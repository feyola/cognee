"""Adapter for Generic API LLM provider API"""

import asyncio
import base64
import logging
import mimetypes
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterable
from copy import deepcopy
from typing import Any

import instructor
import litellm
from instructor.core import InstructorRetryException
from litellm.exceptions import ContentPolicyViolationError
from openai import ContentFilterFinishReasonError
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from cognee.infrastructure.llm.retry_config import (
    llm_retry_condition,
    llm_retry_stop_condition,
)

from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.instructor_modes import (
    get_instructor_mode,
)
from cognee.infrastructure.llm.exceptions import (
    ContentPolicyFilterError,
    LLMPaymentRequiredError,
    is_budget_exhausted_error,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.llm_interface import (
    LLMInterface,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.types import (
    TranscriptionReturnType,
)
from cognee.modules.observability.get_observe import get_observe
from cognee.shared.logging_utils import get_logger
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

logger = get_logger()
observe = get_observe()
_llm_concurrency_semaphores: dict[tuple[int, int], asyncio.Semaphore] = {}


def _get_llm_max_concurrency() -> int:
    raw_value = os.getenv("COGNEE_LLM_MAX_CONCURRENCY") or os.getenv("LLM_MAX_CONCURRENCY", "0")
    try:
        return max(0, int(raw_value))
    except ValueError:
        logger.warning("Invalid LLM concurrency limit %r; disabling the limit.", raw_value)
        return 0


@asynccontextmanager
async def _llm_concurrency_context():
    max_concurrency = _get_llm_max_concurrency()
    if max_concurrency <= 0:
        yield
        return

    loop = asyncio.get_running_loop()
    key = (id(loop), max_concurrency)
    semaphore = _llm_concurrency_semaphores.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(max_concurrency)
        _llm_concurrency_semaphores[key] = semaphore

    async with semaphore:
        yield


def _enrich_llm_span(model: str, name: str) -> None:
    """Set LLM attributes on the current OTEL span, if tracing is enabled."""
    from cognee.modules.observability.trace_context import is_tracing_enabled

    if not is_tracing_enabled():
        return

    try:
        from opentelemetry import trace as otel_trace  # ty:ignore[unresolved-import]

        from cognee.context_global_variables import current_pipeline_stage
        from cognee.modules.observability.tracing import (
            COGNEE_LLM_MODEL,
            COGNEE_LLM_PROVIDER,
            COGNEE_PIPELINE_STAGE,
        )

        current_span = otel_trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute(COGNEE_LLM_MODEL, model)
            current_span.set_attribute(COGNEE_LLM_PROVIDER, name)
            stage = current_pipeline_stage.get()
            if stage:
                current_span.set_attribute(COGNEE_PIPELINE_STAGE, stage)
    except Exception:
        pass


def _copy_reasoning_content_to_empty_content(response: Any) -> Any:
    """Expose local-server structured output returned as reasoning content."""
    try:
        for choice in response.choices or []:
            message = getattr(choice, "message", None)
            if message is None or getattr(message, "content", None):
                continue

            reasoning_content = getattr(message, "reasoning_content", None)
            if not reasoning_content:
                provider_fields = getattr(message, "provider_specific_fields", None) or {}
                reasoning_content = provider_fields.get("reasoning_content")

            if reasoning_content:
                message.content = reasoning_content
    except Exception:
        logger.debug("Could not normalize reasoning_content response field.", exc_info=True)

    return response


async def _materialize_streaming_response(
    response: Any, *, messages: list[dict[str, Any]] | None = None
) -> Any:
    """Combine a LiteLLM async stream into the response expected by Instructor."""
    if not isinstance(response, AsyncIterable):
        return response

    chunks = [chunk async for chunk in response]
    materialized = litellm.stream_chunk_builder(chunks=chunks, messages=messages)
    if materialized is None:
        raise RuntimeError("LLM stream completed without response chunks.")

    return materialized


def _enforce_strict_json_schema(response_format: Any) -> Any:
    """Return a strict OpenAI JSON schema without mutating Instructor's input."""
    if not isinstance(response_format, dict) or response_format.get("type") != "json_schema":
        return response_format

    strict_response_format = deepcopy(response_format)
    schema = strict_response_format.get("json_schema", {}).get("schema")
    if not isinstance(schema, dict):
        return response_format

    def normalize_schema(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                normalize_schema(item)
            return
        if not isinstance(value, dict):
            return

        # OpenAI strict structured outputs support anyOf, but reject the
        # oneOf/discriminator pair emitted for Pydantic discriminated unions.
        # Literal discriminator fields still make the alternatives exclusive,
        # and Pydantic validates the materialized response after generation.
        if "discriminator" in value:
            if "oneOf" in value:
                if "anyOf" in value:
                    raise ValueError(
                        "Cannot normalize a discriminated union containing both oneOf and anyOf."
                    )
                value["anyOf"] = value.pop("oneOf")
            value.pop("discriminator")

        properties = value.get("properties")
        if value.get("type") == "object" or isinstance(properties, dict):
            value["additionalProperties"] = False
            if isinstance(properties, dict):
                value["required"] = list(properties)

        for nested_value in value.values():
            normalize_schema(nested_value)

    normalize_schema(schema)
    return strict_response_format


class GenericAPIAdapter(LLMInterface):
    """
    Adapter for Generic API LLM provider API.

    This class initializes the API adapter with necessary credentials and configurations for
    interacting with a language model. It provides methods for creating structured outputs
    based on user input and system prompts.

    Public methods:
    - acreate_structured_output(text_input: str, system_prompt: str, response_model:
    Type[BaseModel]) -> BaseModel
    """

    MAX_RETRIES = 2
    default_instructor_mode = get_instructor_mode("generic")

    def __init__(
        self,
        api_key: str,
        model: str,
        max_completion_tokens: int,
        name: str,
        endpoint: str | None = None,
        api_version: str | None = None,
        transcription_model: str | None = None,
        image_transcribe_model: str | None = None,
        instructor_mode: str | None = None,
        fallback_model: str | None = None,
        fallback_api_key: str | None = None,
        fallback_endpoint: str | None = None,
        llm_args: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.api_key = api_key
        self.api_version = api_version
        self.endpoint = endpoint
        self.max_completion_tokens = max_completion_tokens
        self.transcription_model = transcription_model or model
        self.image_transcribe_model = image_transcribe_model or model
        self.fallback_model = fallback_model
        self.fallback_api_key = fallback_api_key
        self.fallback_endpoint = fallback_endpoint
        self._base_llm_args: dict[str, Any] = dict(llm_args or {})
        self.enforce_strict_json_schema = bool(
            self._base_llm_args.pop("enforce_strict_json_schema", False)
        )
        self.llm_args = self._base_llm_args

        self.instructor_mode = instructor_mode if instructor_mode else self.default_instructor_mode

        self.aclient = instructor.from_litellm(
            self._acompletion_with_reasoning_content_fallback,
            mode=instructor.Mode(self.instructor_mode),
        )

    async def _acompletion_with_reasoning_content_fallback(self, *args: Any, **kwargs: Any) -> Any:
        if self.enforce_strict_json_schema and "response_format" in kwargs:
            kwargs["response_format"] = _enforce_strict_json_schema(kwargs["response_format"])
        response = await litellm.acompletion(*args, **kwargs)
        response = await _materialize_streaming_response(response, messages=kwargs.get("messages"))
        return _copy_reasoning_content_to_empty_content(response)

    async def acreate_str_output(
        self, text_input: str, system_prompt: str, **merged_kwargs: Any
    ) -> str:
        """Plain-text completion that skips instructor.

        Instructor wraps the call in JSON/tool-call schemas that local
        llama.cpp-compatible servers don't honour, causing repeated parse
        failures and retry storms. A plain string needs no schema, so call
        litellm directly using this adapter's own connection config.
        """
        async with _llm_concurrency_context():
            async with llm_rate_limiter_context_manager():
                response = await litellm.acompletion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text_input},
                    ],
                    api_key=self.api_key,
                    api_base=self.endpoint,
                    api_version=self.api_version,
                    **merged_kwargs,
                )
        response = await _materialize_streaming_response(
            response,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_input},
            ],
        )
        response = _copy_reasoning_content_to_empty_content(response)
        return response.choices[0].message.content or ""

    @observe(as_type="generation")
    @retry(
        stop=llm_retry_stop_condition,
        wait=wait_exponential_jitter(8, 128),
        retry=llm_retry_condition,
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def acreate_structured_output(
        self,
        text_input: str,
        system_prompt: str,
        response_model: type[BaseModel | str],
        **kwargs: Any,
    ) -> BaseModel | str:
        """
        Generate a response from a user query.

        This asynchronous method sends a user query and a system prompt to a language model and
        retrieves the generated response. It handles API communication and retries up to a
        specified limit in case of request failures.

        Parameters:
        -----------

            - text_input (str): The input text from the user to generate a response for.
            - system_prompt (str): A prompt that provides context or instructions for the
              response generation.
            - response_model (Type[BaseModel]): A Pydantic model that defines the structure of
              the expected response.

        Returns:
        --------

            - BaseModel: An instance of the specified response model containing the structured
              output from the language model.
        """

        merged_kwargs = {**self.llm_args, **kwargs}

        # A plain string needs no schema — skip instructor (see acreate_str_output).
        if response_model is str:
            return await self.acreate_str_output(text_input, system_prompt, **merged_kwargs)

        try:
            async with _llm_concurrency_context():
                async with llm_rate_limiter_context_manager():
                    result = await self.aclient.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": f"""{text_input}""",
                            },
                        ],
                        max_retries=self.MAX_RETRIES,
                        api_key=self.api_key,
                        api_base=self.endpoint,
                        response_model=response_model,
                        **merged_kwargs,
                    )
                _enrich_llm_span(self.model, self.name)
                return result
        except (
            ContentFilterFinishReasonError,
            ContentPolicyViolationError,
            InstructorRetryException,
        ) as error:
            if (
                isinstance(error, InstructorRetryException)
                and "content management policy" not in str(error).lower()
            ):
                raise error

            if not (self.fallback_model and self.fallback_api_key and self.fallback_endpoint):
                raise ContentPolicyFilterError(
                    f"The provided input contains content that is not aligned with our content policy: {text_input}"
                ) from error

            fallback_model = self.fallback_model
            fallback_llm_args = {**self._base_llm_args, **kwargs}

            try:
                async with _llm_concurrency_context():
                    async with llm_rate_limiter_context_manager():
                        return await self.aclient.chat.completions.create(
                            model=fallback_model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": system_prompt,
                                },
                                {
                                    "role": "user",
                                    "content": f"""{text_input}""",
                                },
                            ],
                            max_retries=self.MAX_RETRIES,
                            api_key=self.fallback_api_key,
                            api_base=self.fallback_endpoint,
                            response_model=response_model,
                            **fallback_llm_args,
                        )
            except (
                ContentFilterFinishReasonError,
                ContentPolicyViolationError,
                InstructorRetryException,
            ) as error:
                if (
                    isinstance(error, InstructorRetryException)
                    and "content management policy" not in str(error).lower()
                ):
                    raise error
                else:
                    raise ContentPolicyFilterError(
                        f"The provided input contains content that is not aligned with our content policy: {text_input}"
                    ) from error
        except Exception as e:
            if is_budget_exhausted_error(e):
                raise LLMPaymentRequiredError() from e
            raise

    @observe(as_type="transcription")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(2, 128),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.NotFoundError,
                litellm.exceptions.AuthenticationError,
                asyncio.CancelledError,
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def create_transcript(self, input: str, **kwargs: Any) -> TranscriptionReturnType:
        """
        Generate an audio transcript from a user query.

        This method creates a transcript from the specified audio file, raising a
        FileNotFoundError if the file does not exist. The audio file is processed and the
        transcription is retrieved from the API.

        Parameters:
        -----------
            - input: The path to the audio file that needs to be transcribed.

        Returns:
        --------
            The generated transcription of the audio file.
        """
        async with open_data_file(input, mode="rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(input)
        if not mime_type or not mime_type.startswith("audio/"):
            raise ValueError(
                f"Could not determine MIME type for audio file: {input}. Is the extension correct?"
            )
        response = await litellm.acompletion(
            model=self.transcription_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {"file_data": f"data:{mime_type};base64,{encoded_string}"},
                        },
                        {"type": "text", "text": "Transcribe the following audio precisely."},
                    ],
                }
            ],
            api_key=self.api_key,
            api_version=self.api_version,
            max_completion_tokens=self.max_completion_tokens,
            api_base=self.endpoint,
            max_retries=self.MAX_RETRIES,
        )

        if not response.choices or response.choices[0].message is None:
            raise ValueError("Transcription failed. No response received.")
        return TranscriptionReturnType(response.choices[0].message.content, response)

    @observe(as_type="transcribe_image")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(2, 128),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.NotFoundError,
                litellm.exceptions.AuthenticationError,
                asyncio.CancelledError,
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def transcribe_image(
        self,
        input: str,
        prompt: str | None = None,
        max_completion_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> litellm.ModelResponse:
        """
        Generate a transcription of an image from a user query.

        This method encodes the image and sends a request to the API to obtain a
        description of the contents of the image.

        Parameters:
        -----------
            - input: The path to the image file that needs to be transcribed.
            - prompt: Optional extraction instruction; falls back to "What's in this image?".
            - max_completion_tokens: Optional length cap; falls back to 300 when omitted.
            - reasoning_effort: Optional reasoning-effort hint; dropped on models without reasoning.

        Returns:
        --------
            - BaseModel: A structured output generated by the model, returned as an instance of
              BaseModel.
        """
        async with open_data_file(input, mode="rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(input)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(
                f"Could not determine MIME type for image file: {input}. Is the extension correct?"
            )
        response: litellm.ModelResponse = await litellm.acompletion(
            model=self.image_transcribe_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt or "What's in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_image}",
                            },
                        },
                    ],
                }
            ],
            api_key=self.api_key,
            api_base=self.endpoint,
            api_version=self.api_version,
            max_completion_tokens=max_completion_tokens or 300,
            max_retries=self.MAX_RETRIES,
            # drop_params ignores reasoning_effort on models that don't support it.
            reasoning_effort=reasoning_effort,
            drop_params=True,
        )
        return response

import logging
from typing import Any

import anthropic  # ty:ignore[unresolved-import]
import instructor
from instructor.core.patch import AsyncInstructorChatCompletionCreate
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    wait_exponential_jitter,
)

from cognee.infrastructure.llm.retry_config import (
    llm_retry_condition,
    llm_retry_stop_condition,
)

from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.instructor_modes import (
    get_instructor_mode,
)
from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError, is_budget_exhausted_error
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    GenericAPIAdapter,
)
from cognee.modules.observability.get_observe import get_observe
from cognee.shared.logging_utils import get_logger
from cognee.shared.rate_limiting import llm_rate_limiter_context_manager

logger = get_logger()
observe = get_observe()

# Sampling parameters the Anthropic SDK 1.x no longer accepts as keyword
# arguments. ``messages.create`` is generated from the OpenAPI spec, so an
# untyped keyword raises TypeError client-side, before any request is made.
# The REST API still honours all three in the request body, and every adapter
# receives them through ``llm_args`` (LLMConfig folds LLM_TEMPERATURE / LLM_SEED
# in there), so route them into ``extra_body`` rather than letting them explode.
# Dropping them instead would be worse than it looks: extraction would silently
# run at the model default rather than the deterministic temperature the
# operator configured.
BODY_ONLY_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")


def route_body_only_params(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Move sampling parameters the SDK does not type into ``extra_body``.

    Parameters:
    -----------

        - kwargs (dict[str, Any]): Completion keyword arguments; mutated in place.

    Returns:
    --------

        - dict[str, Any]: The same mapping, with sampling parameters relocated.
    """
    routed = {key: kwargs.pop(key) for key in BODY_ONLY_SAMPLING_PARAMS if key in kwargs}
    if routed:
        # An explicit extra_body wins: that is a caller reaching past this shim.
        kwargs["extra_body"] = {**routed, **(kwargs.get("extra_body") or {})}
    return kwargs


class AnthropicAdapter(GenericAPIAdapter):
    """
    Adapter for interfacing with the Anthropic API, enabling structured output generation
    and prompt display.
    """

    # Declared False even though GenericAPIAdapter declares True. This class
    # overrides acreate_structured_output without a `response_model is str`
    # branch and defines no acreate_str_output, so a plain-text answer never
    # reaches the parent's streaming door. Inheriting True would promote a
    # sink, announce `stage: generating`, and then emit nothing at all.
    supports_answer_streaming = False

    default_instructor_mode = get_instructor_mode("anthropic")

    def __init__(
        self,
        api_key: str,
        model: str,
        max_completion_tokens: int,
        instructor_mode: str | None = None,
        llm_args: dict[str, Any] | None = None,
    ) -> None:
        # Support both "model" and "anthropic/model" formats for model names, stripping the "anthropic/" prefix
        model = model.removeprefix("anthropic/") if model.startswith("anthropic/") else model
        super().__init__(
            api_key=api_key,
            model=model,
            max_completion_tokens=max_completion_tokens,
            name="Anthropic",
            llm_args=llm_args,
        )
        self.llm_args: dict[str, Any] = llm_args or {}
        # Anthropic's messages.create requires max_tokens. Unlike the litellm path used by
        # GenericAPIAdapter, instructor.patch around the raw Anthropic SDK does not auto-inject
        # it, so we surface max_completion_tokens here.
        self.llm_args.setdefault("max_tokens", max_completion_tokens)
        self.instructor_mode = instructor_mode if instructor_mode else self.default_instructor_mode

        self.aclient: AsyncInstructorChatCompletionCreate = instructor.patch(
            create=anthropic.AsyncAnthropic(
                api_key=self.api_key,
                http_client=anthropic.DefaultAsyncHttpxClient(http2=False),
            ).messages.create,
            mode=instructor.Mode(self.instructor_mode),
        )

    @observe(as_type="generation")
    @retry(
        stop=llm_retry_stop_condition,
        wait=wait_exponential_jitter(8, 128),
        retry=llm_retry_condition,
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: type[BaseModel], **kwargs: Any
    ) -> BaseModel:
        """
        Generate a response from a user query.

        Parameters:
        -----------

            - text_input (str): The input text from the user to be processed.
            - system_prompt (str): A prompt that sets the context for the query.
            - response_model (Type[BaseModel]): The model to structure the response according to
              its format.

        Returns:
        --------

            - BaseModel: An instance of BaseModel containing the structured response.
        """
        merged_kwargs = route_body_only_params({**self.llm_args, **kwargs})
        try:
            async with llm_rate_limiter_context_manager():
                return await self.aclient(
                    model=self.model,
                    max_retries=2,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Use the given format to extract information
                    from the following input: {text_input}. {system_prompt}""",
                        }
                    ],
                    response_model=response_model,
                    **merged_kwargs,
                )
        except Exception as e:
            if is_budget_exhausted_error(e):
                raise LLMPaymentRequiredError() from e
            raise

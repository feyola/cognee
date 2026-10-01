from importlib import import_module

import pytest

engine_module = import_module(
    "cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine"
)


@pytest.mark.parametrize("override", [None, "", "Qwen/Qwen3-Embedding-4B"])
def test_explicit_tokenizer_is_forwarded_without_changing_model(monkeypatch, override):
    if override is None:
        monkeypatch.delenv("HUGGINGFACE_TOKENIZER", raising=False)
    else:
        monkeypatch.setenv("HUGGINGFACE_TOKENIZER", override)
    calls = []
    resolved = object()

    def resolve(**kwargs):
        calls.append(kwargs)
        return resolved

    monkeypatch.setattr(engine_module, "resolve_embedding_tokenizer", resolve)
    engine = object.__new__(engine_module.OpenAICompatibleEmbeddingEngine)
    engine.model = "served-model.gguf"

    assert engine.get_tokenizer() is resolved
    assert calls == [
        {
            "provider": "openai_compatible",
            "model": "served-model.gguf",
            "huggingface_tokenizer": override or None,
        }
    ]

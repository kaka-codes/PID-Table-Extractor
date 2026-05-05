import os
from functools import lru_cache
from typing import Any, Dict, Iterable

from llm.prompt import build_qa_prompt


DEFAULT_MODEL_PATH = os.getenv(
    "PID_QA_MODEL_PATH",
    "C:/Users/INT-Viveck/Downloads/Phi-3-mini-4k-instruct-q4.gguf",
)


@lru_cache(maxsize=4)
def load_model(model_path: str = DEFAULT_MODEL_PATH, n_ctx: int = 2048, n_threads: int = 6):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "Local model loading requires 'llama-cpp-python'. Install it to run QA."
        ) from exc

    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False,
    )


def ask_llm(
    llm: Any,
    question: str,
    retrieved_chunks: Iterable[Dict[str, Any]],
    metadata: Dict[str, Any],
    max_tokens: int = 256,
) -> str:
    prompt = build_qa_prompt(
        question=question,
        retrieved_chunks=list(retrieved_chunks),
        metadata=metadata,
    )

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        stop=["\nQuestion:", "\n\n\n"],
    )

    return output["choices"][0]["text"].strip()

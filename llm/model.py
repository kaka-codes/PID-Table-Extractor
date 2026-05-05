import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from llm.prompt import build_qa_prompt


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STREAMLIT_SCRIPT = APP_ROOT / "app.py"
DEFAULT_SHARED_VENV_PYTHON = APP_ROOT.parent / "venv" / "Scripts" / "python.exe"
DEFAULT_RUN_SCRIPT = APP_ROOT / "run_ui.ps1"
DEFAULT_MODEL_PATH = os.getenv(
    "PID_QA_MODEL_PATH",
    "C:/Users/INT-Viveck/Downloads/Phi-3-mini-4k-instruct-q4.gguf",
)


def _recommended_python_path() -> Optional[Path]:
    configured_python = os.getenv("PID_QA_PYTHON")
    if configured_python:
        candidate = Path(configured_python).expanduser()
        if candidate.exists():
            return candidate

    if DEFAULT_SHARED_VENV_PYTHON.exists():
        return DEFAULT_SHARED_VENV_PYTHON

    return None


def get_local_model_runtime_status() -> tuple[bool, str]:
    try:
        from llama_cpp import Llama  # noqa: F401
    except ImportError:
        command = None
        recommended_python = _recommended_python_path()

        if DEFAULT_RUN_SCRIPT.exists():
            command = f'powershell -ExecutionPolicy Bypass -File "{DEFAULT_RUN_SCRIPT}"'
        elif recommended_python is not None:
            command = f'"{recommended_python}" -m streamlit run "{DEFAULT_STREAMLIT_SCRIPT}"'

        message_parts = [
            "Local QA is unavailable because 'llama-cpp-python' is not installed in the current Python environment.",
            f"Current Python: {sys.executable}",
        ]

        if command:
            message_parts.append(f"Start the UI with: {command}")
        else:
            message_parts.append(
                "Install 'llama-cpp-python' in the Python environment that launches Streamlit."
            )

        return False, " ".join(message_parts)

    return True, ""


@lru_cache(maxsize=4)
def load_model(model_path: str = DEFAULT_MODEL_PATH, n_ctx: int = 4096, n_threads: int = 6):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        _, message = get_local_model_runtime_status()
        raise RuntimeError(message) from exc

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

    if hasattr(llm, "create_chat_completion"):
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return output["choices"][0]["message"]["content"].strip()

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        stop=["\nQuestion:", "\n\n\n"],
    )

    return output["choices"][0]["text"].strip()

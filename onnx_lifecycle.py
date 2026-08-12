"""ONNX Runtime provider policy and session lifecycle helpers."""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import onnxruntime as ort

from .utils import model_url


ONNX_PROVIDER_POLICY_ENV = "COMFYUI_HUMAN_PARTS_ONNX_PROVIDER"
ONNX_PROVIDER_POLICIES = {
    "auto": None,
    "tensorrt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
ONNX_PROVIDER_PRIORITY = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


def lifecycle_execution_providers(policy: str | None = None) -> tuple[str, ...]:
    """Resolve advertised providers according to an explicit preference policy.

    ``auto`` tries accelerated providers in priority order and retains all
    advertised lower-priority providers as fallbacks. An explicit policy starts
    at that provider (for example, ``cuda`` means CUDA then CPU).
    """
    selected_policy = (
        policy or os.getenv(ONNX_PROVIDER_POLICY_ENV, "auto")
    ).strip().lower()
    if selected_policy not in ONNX_PROVIDER_POLICIES:
        choices = ", ".join(ONNX_PROVIDER_POLICIES)
        raise ValueError(
            f"Invalid ONNX provider policy {selected_policy!r}. "
            f"Choose one of: {choices}."
        )

    available_list = ort.get_available_providers()
    available = set(available_list)
    requested = ONNX_PROVIDER_POLICIES[selected_policy]
    if requested is not None and requested not in available:
        advertised = ", ".join(available_list) or "none"
        raise RuntimeError(
            f"ONNX provider policy {selected_policy!r} requires {requested}, "
            f"but ONNX Runtime advertises: {advertised}. Use "
            f"{ONNX_PROVIDER_POLICY_ENV}=auto or install the matching "
            "ONNX Runtime package and system libraries."
        )

    start = 0 if requested is None else ONNX_PROVIDER_PRIORITY.index(requested)
    providers = tuple(
        provider
        for provider in ONNX_PROVIDER_PRIORITY[start:]
        if provider in available
    )
    # Preserve support for other ONNX Runtime builds (CoreML, DirectML, etc.)
    # when none of the providers in our normal priority list is advertised.
    return providers or tuple(available_list)


def execution_providers(policy: str | None = None) -> tuple[str, ...]:
    """Backward-compatible short name for the provider policy resolver."""
    return lifecycle_execution_providers(policy)


@lru_cache(maxsize=1)
def load_session(path: str, providers: tuple[str, ...]) -> ort.InferenceSession:
    """Load an ONNX model, dropping providers that fail initialization."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Human Parts ONNX model not found at {path}. Run install.py with "
            "the same Python environment that starts ComfyUI "
            f"(`{sys.executable} install.py`), or download "
            f"{model_url} and save it at that exact path."
        )

    # Keep CPU behind CUDA so unsupported operators may fall back to CPU. If a
    # provider cannot initialize at all, drop it from the next session attempt.
    attempts = [providers[index:] for index in range(len(providers))]
    if not attempts:
        attempts.append(())

    errors: list[tuple[tuple[str, ...], Exception]] = []
    for provider_attempt in attempts:
        try:
            if provider_attempt:
                return ort.InferenceSession(path, providers=list(provider_attempt))
            return ort.InferenceSession(path)
        except Exception as error:  # ONNX Runtime uses several exception types.
            errors.append((provider_attempt, error))

    failures = "; ".join(
        f"{'+'.join(provider_attempt) or 'ONNX Runtime default'}: "
        f"{type(error).__name__}: {error}"
        for provider_attempt, error in errors
    )
    raise RuntimeError(
        f"Unable to load Human Parts ONNX model at {path}. "
        f"Provider failures: {failures}"
    ) from errors[-1][1]

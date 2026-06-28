import os


_SECRET_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


_RAY_ENV_PREFIXES = (
    "SGLANG_",
    "NCCL_",
    "TORCH_NCCL_",
)


_RAY_ENV_NAMES = {
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_ENABLE_COREDUMP_ON_EXCEPTION",
    "CUDA_COREDUMP_SHOW_PROGRESS",
    "CUDA_COREDUMP_GENERATION_FLAGS",
    "CUDA_COREDUMP_FILE",
    "GLOO_SOCKET_IFNAME",
    "INDEXER_ROPE_NEOX_STYLE",
    "MASTER_ADDR",
    "MC_IB_PCI_RELAXED_ORDERING",
    "MLP_SKIP_SORT_RDMA",
    "NVSHMEM_DISABLE_NCCL",
    "PYTHONUNBUFFERED",
    "RAY_memory_monitor_refresh_ms",
    "RAY_memory_usage_threshold",
    "TP_SOCKET_IFNAME",
}


def _split_env_names(raw: str | None) -> set[str]:
    if not raw:
        return set()
    names = set()
    for item in raw.replace(",", " ").split():
        item = item.strip()
        if item:
            names.add(item)
    return names


def _looks_secret(name: str) -> bool:
    upper = name.upper()
    return any(part in upper for part in _SECRET_NAME_PARTS)


def collect_ray_runtime_env_passthrough(env: dict[str, str] | None = None) -> dict[str, str]:
    """Collect non-secret runtime env vars that must survive Ray process hops."""

    source = os.environ if env is None else env
    explicit_names = _split_env_names(source.get("SLIME_RAY_RUNTIME_ENV_PASSTHROUGH"))
    names = set(_RAY_ENV_NAMES) | explicit_names
    for name in source:
        if any(name.startswith(prefix) for prefix in _RAY_ENV_PREFIXES):
            names.add(name)

    result = {}
    for name in sorted(names):
        if _looks_secret(name):
            continue
        value = source.get(name)
        if value is not None:
            result[name] = str(value)
    return result


def format_env_key_list(env_vars: dict[str, str]) -> str:
    return ",".join(sorted(env_vars))

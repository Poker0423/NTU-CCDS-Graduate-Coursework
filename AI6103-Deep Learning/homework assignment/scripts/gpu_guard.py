"""GPU 6 only. Standard-library launch checks run BEFORE torch imports."""
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone

GPU6_UUID = os.environ.get("AI6103_GPU6_UUID", "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")  # set for your own server
HOST = os.environ.get("AI6103_GPU6_HOST", "your-gpu-server")
REMOTE_BASE = os.environ.get("AI6103_REMOTE_BASE", "/path/to/ai6103-hw")
MIN_FREE_MIB = 32768
MAX_IDLE_UTIL = 10
MAX_OWN_RESERVED = 24 * 1024**3
_identity = None


class GPUAccessError(RuntimeError):
    pass


def on_server():
    return platform.node().split(".")[0] == HOST


def bootstrap(env=None):
    """No CUDA calls, no imports of torch, and no automatic environment repair."""
    env = os.environ if env is None else env
    profile = env.get("AI6103_PROFILE")
    if profile == "cpu":
        if env.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise GPUAccessError("Explicit CPU QA requires CUDA_VISIBLE_DEVICES=-1")
        return "cpu"
    if profile == "tc2":
        try:
            from . import tc2_guard
        except ImportError:
            import tc2_guard
        tc2_guard.bootstrap(env)
        return "tc2"
    if profile == "gpu6" or on_server():
        if profile != "gpu6" or env.get("CUDA_VISIBLE_DEVICES") != GPU6_UUID:
            raise GPUAccessError("Explicit gpu6 profile and exact full GPU 6 UUID required; no fallback")
        if env.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            raise GPUAccessError("CUDA_DEVICE_ORDER=PCI_BUS_ID required")
        if env.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise GPUAccessError("Deterministic CUBLAS configuration required before CUDA imports")
        return "gpu6"
    if profile not in (None, ""):
        raise GPUAccessError("Unknown AI6103 profile")
    return "local"


def query_gpu6(require_idle=False, min_free_mib=MIN_FREE_MIB):
    bootstrap()
    result = subprocess.run(["nvidia-smi", "-i", GPU6_UUID,
        "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise GPUAccessError("GPU 6 identity query did not return one device")
    fields = [s.strip() for s in lines[0].split(",")]
    if len(fields) != 7 or fields[0] != "6" or fields[1] != GPU6_UUID or "H200" not in fields[2]:
        raise GPUAccessError("Physical GPU index/UUID/model mismatch")
    total, free, util = map(int, fields[3:6])
    driver = tuple(int(x) for x in fields[6].split("."))
    if driver < (550, 54, 14):
        raise GPUAccessError("Driver insufficient for planned CUDA 12.4 environment")
    if free < min_free_mib or (require_idle and util > MAX_IDLE_UTIL):
        raise GPUAccessError(f"GPU 6 busy/low memory: free={free} MiB util={util}%; stop, no other GPU")
    return {"physical_index": 6, "uuid": GPU6_UUID, "name": fields[2], "total_mib": total,
            "free_mib": free, "utilization_percent": util, "driver": fields[6],
            "timestamp": datetime.now(timezone.utc).isoformat()}


def normalize_uuid(value):
    text = str(value)
    if text.startswith("GPU-"):
        text = text[4:]
    try:
        return "GPU-" + str(uuid.UUID(text))
    except (ValueError, AttributeError):
        raise GPUAccessError("CUDA device UUID is unavailable/unverifiable")


def verify_cuda(torch_module, require_idle=False):
    global _identity
    profile = bootstrap()
    if profile == "tc2":
        try:
            from . import tc2_guard
        except ImportError:
            import tc2_guard
        _identity = tc2_guard.verify_cuda(torch_module)
        return _identity.copy(), {key: _identity[key] for key in
                                  ("physical_gpu_index", "physical_gpu_uuid", "gpu_name",
                                   "gpu_total_mib", "gpu_free_mib", "gpu_utilization_percent", "nvidia_driver")}
    if profile != "gpu6":
        raise GPUAccessError("GPU execution requires explicit GPU 6 profile even outside the server")
    # NVML metadata must pass before torch's first CUDA availability/context operation.
    snapshot = query_gpu6(require_idle=require_idle)
    cuda = torch_module.cuda
    if not cuda.is_available() or cuda.device_count() != 1:
        raise GPUAccessError("Exactly one available masked CUDA device required")
    props = cuda.get_device_properties(0)
    actual = normalize_uuid(getattr(props, "uuid", None))
    if actual != GPU6_UUID:
        raise GPUAccessError("Runtime CUDA UUID does not match authorized physical GPU 6")
    cuda.set_device(0)
    cuda.set_per_process_memory_fraction(min(1., MAX_OWN_RESERVED / props.total_memory), 0)
    free, total = cuda.mem_get_info(0)
    if free < MIN_FREE_MIB * 1024**2:
        raise GPUAccessError("Insufficient current free VRAM on the selected card")
    _identity = {"physical_gpu_index": 6, "physical_gpu_uuid": actual, "logical_gpu_index": 0,
                 "cuda_visible_devices": GPU6_UUID, "cuda_device_order": "PCI_BUS_ID",
                 "nvidia_driver": snapshot["driver"], "policy": "GPU6_ONLY_SINGLE_DEVICE"}
    return _identity.copy(), snapshot


def identity():
    if _identity is None:
        raise GPUAccessError("CUDA identity not verified")
    bootstrap()
    return _identity.copy()


def check_running_resources(torch_module):
    """No total utilization test during OUR computation; preserve room for other users."""
    profile = bootstrap()
    if profile == "tc2":
        try:
            from . import tc2_guard
        except ImportError:
            import tc2_guard
        return tc2_guard.check_running_resources(torch_module, identity())
    snapshot = query_gpu6(min_free_mib=16384)
    if torch_module.cuda.memory_reserved(0) > MAX_OWN_RESERVED:
        raise GPUAccessError("Our CUDA reservation exceeded the 24 GiB conservative cap")
    return snapshot

"""Fail-closed NTU TC2 SLURM allocation guard; no torch import at module load."""
import os
import re
import shlex
import socket
import subprocess
import uuid

PARTITION = "MGPU-TC2"
QOS = "normal"
MAX_GPUS = 1
MAX_CPUS = 10
MAX_MEMORY_MIB = 30 * 1024
MAX_WALL_SECONDS = 6 * 60 * 60
MAX_OWN_RESERVED = 12 * 1024**3
MIN_FREE_MIB = 6 * 1024


class TC2AccessError(RuntimeError):
    pass


def _single(value, name):
    items = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if len(items) != 1 or items[0] in ("-1", "NoDevFiles"):
        raise TC2AccessError(f"Exactly one {name} entry is required")
    return items[0]


def _parse_scontrol(text):
    record = {}
    for token in shlex.split(text.strip()):
        if "=" in token:
            key, value = token.split("=", 1)
            record[key] = value
    return record


def _gpu_counts(record):
    values = []
    for key in ("TresPerNode", "AllocTRES", "TRES", "Gres"):
        value = record.get(key, "")
        for pattern in (r"(?:^|,)gres/gpu(?:[^=,]*)=(\d+)(?:,|$)",
                        r"(?:^|,)gres:gpu(?:[^:,()]*)?:(\d+)(?:\(|,|$)",
                        r"(?:^|,)gpu(?:[^=,]*)=(\d+)(?:,|$)"):
            values.extend(int(match) for match in re.findall(pattern, value))
    return values


def _wall_seconds(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", str(value or ""))
    if not match:
        raise TC2AccessError("SLURM TimeLimit is missing or ambiguous")
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def slurm_allocation(env=None, runner=subprocess.run):
    """Validate scheduler facts without importing/initializing CUDA."""
    env = os.environ if env is None else env
    if env.get("AI6103_PROFILE") != "tc2":
        raise TC2AccessError("Explicit AI6103_PROFILE=tc2 is required")
    job_id = str(env.get("SLURM_JOB_ID", ""))
    if not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id):
        raise TC2AccessError("A valid SLURM_JOB_ID is required")
    visible = _single(env.get("CUDA_VISIBLE_DEVICES"), "CUDA-visible GPU")
    allocated = _single(env.get("SLURM_JOB_GPUS"), "SLURM_JOB_GPUS")
    for value in (visible, allocated):
        if not value.isdigit() and not (value.startswith("GPU-") and _normalize_uuid(value) == value):
            raise TC2AccessError("GPU identifiers must be a single ordinal or complete GPU UUID")
    if env.get("SLURM_STEP_GPUS"):
        _single(env["SLURM_STEP_GPUS"], "SLURM_STEP_GPUS")
    if env.get("SLURM_GPUS_ON_NODE") not in (None, "", "1"):
        raise TC2AccessError("SLURM_GPUS_ON_NODE must be exactly one")
    if env.get("SLURM_JOB_NUM_NODES") not in (None, "", "1"):
        raise TC2AccessError("TC2 jobs must use one node")
    node = env.get("SLURMD_NODENAME")
    if not node:
        raise TC2AccessError("SLURMD_NODENAME is required; head-node execution is forbidden")
    result = runner(["scontrol", "show", "job", "-o", job_id], capture_output=True,
                    text=True, check=True, timeout=8)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise TC2AccessError("SLURM job metadata must contain exactly one record")
    record = _parse_scontrol(lines[0])
    if record.get("JobId") != job_id or record.get("Partition") != PARTITION or record.get("QOS") != QOS:
        raise TC2AccessError("SLURM JobId/Partition/QOS does not match the authorized allocation")
    if record.get("JobState") != "RUNNING":
        raise TC2AccessError("The SLURM job is not running")
    counts = _gpu_counts(record)
    if not counts or any(count != 1 for count in counts):
        raise TC2AccessError("Reliable SLURM metadata must prove exactly one allocated GPU")
    if record.get("NodeList") in (None, "", "(null)"):
        raise TC2AccessError("SLURM did not identify an allocated compute node")
    if record["NodeList"] != node:
        raise TC2AccessError("SLURM compute-node identity is ambiguous")
    if socket.gethostname().split(".")[0] != node.split(".")[0]:
        raise TC2AccessError("Current process is not on the allocated compute node")
    if record.get("BatchHost", node) != node:
        raise TC2AccessError("SLURM batch host does not match the current allocation")
    if env.get("SLURM_JOB_PARTITION", PARTITION) != PARTITION or env.get("SLURM_JOB_QOS", QOS) != QOS:
        raise TC2AccessError("Job environment partition/QoS disagrees with scheduler metadata")
    if record.get("UserId") and hasattr(os, "getuid"):
        owner = re.fullmatch(r"[^()]+\((\d+)\)", record["UserId"])
        if not owner or int(owner[1]) != os.getuid():
            raise TC2AccessError("Current process user does not own the SLURM job")
    cpus = int(env.get("SLURM_CPUS_PER_TASK", "0") or 0)
    memory = int(env.get("SLURM_MEM_PER_NODE", "0") or 0)
    if not 1 <= cpus <= MAX_CPUS:
        raise TC2AccessError("Allocated CPU count is missing or exceeds the authorized TC2 QoS")
    if not 1 <= memory <= MAX_MEMORY_MIB:
        raise TC2AccessError("Allocated memory is missing or exceeds the authorized TC2 QoS")
    if record.get("NumNodes", "1") != "1" or not 1 <= int(record.get("NumCPUs", cpus)) <= MAX_CPUS:
        raise TC2AccessError("Scheduler node/CPU allocation exceeds the authorized TC2 QoS")
    allocated_tres = dict(item.split("=", 1) for item in record.get("AllocTRES", "").split(",") if "=" in item)
    if "cpu" in allocated_tres and not 1 <= int(allocated_tres["cpu"]) <= MAX_CPUS:
        raise TC2AccessError("Scheduler AllocTRES CPU allocation exceeds the authorized TC2 QoS")
    if "mem" in allocated_tres:
        amount = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)", allocated_tres["mem"])
        if not amount:
            raise TC2AccessError("Scheduler memory allocation is ambiguous")
        allocated_mib = float(amount[1]) * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2}[amount[2]]
        if not 0 < allocated_mib <= MAX_MEMORY_MIB:
            raise TC2AccessError("Scheduler memory allocation exceeds the authorized TC2 QoS")
    if not 0 < _wall_seconds(record.get("TimeLimit")) <= MAX_WALL_SECONDS:
        raise TC2AccessError("Allocated wall time exceeds the authorized TC2 QoS")
    return {"slurm_job_id": job_id, "slurm_partition": PARTITION, "slurm_qos": QOS,
            "slurm_job_gpus": allocated, "cuda_visible_devices": visible,
            "compute_node": node, "slurm_node_list": record["NodeList"],
            "slurm_cpus_per_task": cpus, "slurm_mem_per_node_mib": memory,
            "slurm_timelimit": record.get("TimeLimit"), "slurm_gres": record.get("TresPerNode", record.get("Gres"))}


def bootstrap(env=None, runner=subprocess.run):
    allocation = slurm_allocation(env, runner)
    return allocation


def _normalize_uuid(value):
    text = str(value)
    if text.startswith("GPU-"):
        text = text[4:]
    try:
        return "GPU-" + str(uuid.UUID(text))
    except (ValueError, AttributeError):
        raise TC2AccessError("CUDA device UUID is unavailable or invalid")


def query_allocated_gpu(allocation, runner=subprocess.run, min_free_mib=MIN_FREE_MIB):
    """Require one job-visible NVIDIA GPU; use its verified UUID for later queries.

    SLURM global IDs and NVIDIA job-visible ordinals are separate namespaces. The
    authorization basis is the validated SLURM job and scheduler-enforced visibility,
    not UUID equality alone. Broader NVIDIA visibility is rejected, never searched.
    """
    expected_uuid = allocation.get("physical_gpu_uuid")
    command = ["nvidia-smi"]
    if expected_uuid is not None:
        if not str(expected_uuid).startswith("GPU-") or _normalize_uuid(expected_uuid) != expected_uuid:
            raise TC2AccessError("A complete previously verified GPU UUID is required")
        command.extend(["-i", expected_uuid])
    command.extend(["--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version",
                    "--format=csv,noheader,nounits"])
    result = runner(command, capture_output=True, text=True,
                    check=True, timeout=8)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise TC2AccessError("NVIDIA visibility must identify exactly one unambiguous device")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 7:
        raise TC2AccessError("Unexpected nvidia-smi response")
    visible_index, gpu_uuid, name = fields[:3]
    if not visible_index.isdigit() or not gpu_uuid.startswith("GPU-") or _normalize_uuid(gpu_uuid) != gpu_uuid:
        raise TC2AccessError("Invalid NVIDIA-visible ordinal or UUID")
    try:
        total, free, utilization = map(int, fields[3:6])
    except ValueError as error:
        raise TC2AccessError("NVIDIA memory/utilization metrics are unavailable") from error
    if not 0 <= free <= total or not 0 <= utilization <= 100:
        raise TC2AccessError("Invalid NVIDIA memory/utilization metrics")
    if expected_uuid is not None and gpu_uuid != expected_uuid:
        raise TC2AccessError("NVIDIA UUID query returned an unexpected GPU")
    if str(allocation["slurm_job_gpus"]).startswith("GPU-") and gpu_uuid != allocation["slurm_job_gpus"]:
        raise TC2AccessError("Scheduler-assigned UUID disagrees with NVIDIA visibility")
    if free < min_free_mib:
        raise TC2AccessError(f"Allocated GPU has insufficient free memory: {free} MiB")
    return {"physical_gpu_index": None, "nvidia_visible_index": int(visible_index), "physical_gpu_uuid": gpu_uuid,
            "gpu_name": name, "gpu_total_mib": total, "gpu_free_mib": free,
            "gpu_utilization_percent": utilization, "nvidia_driver": fields[6]}


def verify_cuda(torch_module, env=None, runner=subprocess.run):
    allocation = bootstrap(env, runner)
    physical = query_allocated_gpu(allocation, runner)
    cuda = torch_module.cuda
    if not cuda.is_available() or cuda.device_count() != 1:
        raise TC2AccessError("PyTorch must see exactly one allocated CUDA device")
    props = cuda.get_device_properties(0)
    runtime_uuid = _normalize_uuid(getattr(props, "uuid", None))
    if runtime_uuid != physical["physical_gpu_uuid"]:
        raise TC2AccessError("PyTorch logical cuda:0 UUID does not match the SLURM-assigned physical GPU")
    physical = query_allocated_gpu({**allocation, **physical}, runner)
    cuda.set_device(0)
    cuda.set_per_process_memory_fraction(min(1.0, MAX_OWN_RESERVED / props.total_memory), 0)
    free_bytes, total_bytes = cuda.mem_get_info(0)
    if free_bytes < MIN_FREE_MIB * 1024**2:
        raise TC2AccessError("Insufficient runtime memory on the allocated GPU")
    return {**allocation, **physical, "logical_gpu_index": 0,
            "runtime_gpu_uuid": runtime_uuid, "runtime_gpu_name": props.name,
            "runtime_gpu_total_bytes": total_bytes, "cuda_runtime": torch_module.version.cuda,
            "torch_version": torch_module.__version__, "policy": "TC2_SLURM_SINGLE_GPU",
            "allocation_authority": "Active SLURM job and scheduler-controlled device visibility; UUID cross-check only",
            "index_semantics": "slurm_job_gpus is scheduler global ID; nvidia_visible_index is driver-visible ordinal; physical index unverified",
            "host": socket.gethostname()}


def check_running_resources(torch_module, identity, runner=subprocess.run, env=None):
    allocation = bootstrap(env, runner)
    for key in ("slurm_job_id", "compute_node", "slurm_job_gpus", "cuda_visible_devices"):
        if allocation[key] != identity[key]:
            raise TC2AccessError("SLURM job/device context changed during execution")
    physical = query_allocated_gpu({**allocation, "physical_gpu_uuid": identity["physical_gpu_uuid"]},
                                  runner, min_free_mib=2048)
    if torch_module.cuda.device_count() != 1 or _normalize_uuid(
            getattr(torch_module.cuda.get_device_properties(0), "uuid", None)) != identity["physical_gpu_uuid"]:
        raise TC2AccessError("Runtime CUDA visibility or UUID changed during execution")
    if torch_module.cuda.memory_reserved(0) > MAX_OWN_RESERVED:
        raise TC2AccessError("This process exceeded the 12 GiB TC2 CUDA reservation cap")
    return physical

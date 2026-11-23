from types import SimpleNamespace
import subprocess
import pytest

from scripts import tc2_guard as guard

GPU_UUID = "GPU-12345678-1234-1234-1234-123456789abc"


@pytest.fixture(autouse=True)
def allocated_process_host(monkeypatch):
    monkeypatch.setattr(guard.socket, "gethostname", lambda: "TC2N03")


def valid_env(**updates):
    env = {"AI6103_PROFILE": "tc2", "SLURM_JOB_ID": "42050",
           "CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "2",
           "SLURM_GPUS_ON_NODE": "1", "SLURM_JOB_NUM_NODES": "1",
           "SLURMD_NODENAME": "TC2N03", "SLURM_CPUS_PER_TASK": "2",
           "SLURM_MEM_PER_NODE": "12288"}
    env.update(updates)
    return env


def runner(record=None, uuid=GPU_UUID, inventory=None, seen=None):
    record = record or ("JobId=42050 JobState=RUNNING Partition=MGPU-TC2 QOS=normal "
                        "NodeList=TC2N03 TresPerNode=gres:gpu:1 AllocTRES=cpu=2,mem=12G,node=1,gres/gpu=1 TimeLimit=00:15:00")
    def run(command, **kwargs):
        if seen is not None:
            seen.append(command)
        assert kwargs["check"] and kwargs["timeout"] == 8
        if command[0] == "scontrol":
            return SimpleNamespace(stdout=record + "\n")
        if "-i" in command:
            # Exact TC2 failure: no code may treat scheduler ID 2 as a driver index.
            if command[2] == "2":
                raise subprocess.CalledProcessError(6, command)
            assert command[2] == GPU_UUID
        output = f"0, {uuid}, NVIDIA L40S, 46068, 40000, 0, 570.148.08\n"
        return SimpleNamespace(stdout=inventory if inventory is not None and "-i" not in command else output)
    return run


def fake_torch(calls, count=1, uuid=GPU_UUID, available=True):
    def event(name, value=None):
        def call(*args):
            calls.append((name, args))
            return value
        return call
    cuda = SimpleNamespace(is_available=event("available", available),
                           device_count=event("count", count),
                           get_device_properties=event("properties", SimpleNamespace(
                               uuid=uuid, name="NVIDIA L40S", total_memory=46 * 1024**3)),
                           set_device=event("set_device"),
                           set_per_process_memory_fraction=event("cap"),
                           memory_reserved=event("reserved", 5 * 1024**3),
                           mem_get_info=event("memory", (40 * 1024**3, 46 * 1024**3)))
    return SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.4"), __version__="2.6.0+cu124")


def test_valid_slurm_remapping_and_cuda_identity():
    calls = []
    identity = guard.verify_cuda(fake_torch(calls), valid_env(), runner())
    assert identity["slurm_job_gpus"] == "2"
    assert identity["cuda_visible_devices"] == "0"
    assert identity["logical_gpu_index"] == 0
    assert identity["physical_gpu_index"] is None and identity["physical_gpu_uuid"] == GPU_UUID
    assert identity["nvidia_visible_index"] == 0
    assert identity["slurm_partition"] == "MGPU-TC2" and identity["slurm_qos"] == "normal"
    assert ("set_device", (0,)) in calls


@pytest.mark.parametrize("updates", [
    {"SLURM_JOB_ID": ""}, {"CUDA_VISIBLE_DEVICES": "0,1"},
    {"SLURM_JOB_GPUS": "2,5"}, {"SLURM_GPUS_ON_NODE": "2"},
    {"SLURMD_NODENAME": ""}, {"AI6103_PROFILE": "gpu6"},
    {"SLURM_CPUS_PER_TASK": "11"}, {"SLURM_MEM_PER_NODE": "30721"},
    {"SLURM_JOB_GPUS": "bogus"}, {"CUDA_VISIBLE_DEVICES": "GPU-invalid"},
    {"SLURM_STEP_GPUS": "2,3"}, {"SLURM_JOB_QOS": "high"},
])
def test_invalid_slurm_environment_fails_before_cuda(updates):
    calls = []
    with pytest.raises(guard.TC2AccessError):
        guard.verify_cuda(fake_torch(calls), valid_env(**updates), runner())
    assert not calls


@pytest.mark.parametrize("change", [
    "JobId=42050 JobState=RUNNING Partition=OTHER QOS=normal NodeList=TC2N03 TresPerNode=gres:gpu:1",
    "JobId=42050 JobState=RUNNING Partition=MGPU-TC2 QOS=high NodeList=TC2N03 TresPerNode=gres:gpu:1",
    "JobId=42050 JobState=PENDING Partition=MGPU-TC2 QOS=normal NodeList=(null) TresPerNode=gres:gpu:1",
    "JobId=42050 JobState=RUNNING Partition=MGPU-TC2 QOS=normal NodeList=TC2N03 TresPerNode=gres:gpu:2",
    "JobId=42050 JobState=COMPLETING Partition=MGPU-TC2 QOS=normal NodeList=TC2N03 TresPerNode=gres:gpu:1 TimeLimit=00:15:00",
    "JobId=42050 JobState=RUNNING Partition=MGPU-TC2 QOS=normal NodeList=TC2N02 TresPerNode=gres:gpu:1 TimeLimit=00:15:00",
    "JobId=42050 JobState=RUNNING Partition=MGPU-TC2 QOS=normal NodeList=TC2N03 TresPerNode=gres:gpu:1 TimeLimit=06:00:01",
])
def test_invalid_scheduler_metadata_fails_closed(change):
    with pytest.raises(guard.TC2AccessError):
        guard.slurm_allocation(valid_env(), runner(change))


@pytest.mark.parametrize("count,uuid,available", [(0, GPU_UUID, True), (2, GPU_UUID, True),
                                                    (1, None, True), (1, "GPU-00000000-0000-0000-0000-000000000000", True),
                                                    (1, GPU_UUID, False)])
def test_cuda_visibility_and_uuid_fail_closed(count, uuid, available):
    calls = []
    with pytest.raises(guard.TC2AccessError):
        guard.verify_cuda(fake_torch(calls, count, uuid, available), valid_env(), runner())
    assert not any(name in ("set_device", "cap") for name, _ in calls)


def test_gpu_queries_use_inventory_then_uuid_never_global_or_guessed_index():
    allocation = guard.slurm_allocation(valid_env(CUDA_VISIBLE_DEVICES="0", SLURM_JOB_GPUS="5"),
                                        runner())
    seen = []
    physical = runner(seen=seen)
    info = guard.query_allocated_gpu(allocation, physical)
    guard.query_allocated_gpu({**allocation, **info}, physical)
    assert "-i" not in seen[0]
    assert seen[1][:3] == ["nvidia-smi", "-i", GPU_UUID]
    assert info["nvidia_visible_index"] == 0 and info["physical_gpu_index"] is None


def test_numeric_selector_failure_is_avoided_by_initial_and_runtime_checks():
    seen, calls = [], []
    run = runner(seen=seen)
    with pytest.raises(subprocess.CalledProcessError) as old:
        run(["nvidia-smi", "-i", "2"], check=True, timeout=8)
    assert old.value.returncode == 6
    seen.clear()
    torch = fake_torch(calls)
    identity = guard.verify_cuda(torch, valid_env(), run)
    snapshot = guard.check_running_resources(torch, identity, run, env=valid_env())
    assert snapshot["physical_gpu_uuid"] == GPU_UUID
    assert all(command[2] == GPU_UUID for command in seen if command[:2] == ["nvidia-smi", "-i"])
    assert ("reserved", (0,)) in calls


@pytest.mark.parametrize("inventory", ["", "0, invalid, NVIDIA A40, 46068, 40000, 0, 570.148.08\n",
    f"0, {GPU_UUID}, NVIDIA A40, 46068, 40000, 0, 570.148.08\n" * 2])
def test_missing_invalid_or_multiple_nvidia_inventory_fails_before_cuda(inventory):
    calls = []
    with pytest.raises(guard.TC2AccessError):
        guard.verify_cuda(fake_torch(calls), valid_env(), runner(inventory=inventory))
    assert calls == []


@pytest.mark.parametrize("kind", ["runtime_uuid", "nvidia_uuid", "count", "cap", "job_context"])
def test_runtime_device_or_context_changes_fail_closed(kind):
    calls = []
    torch = fake_torch(calls)
    identity = guard.verify_cuda(torch, valid_env(), runner())
    env = valid_env()
    run = runner()
    if kind == "runtime_uuid":
        torch.cuda.get_device_properties = lambda index: SimpleNamespace(uuid="GPU-00000000-0000-0000-0000-000000000000")
    elif kind == "nvidia_uuid":
        run = runner(uuid="GPU-00000000-0000-0000-0000-000000000000")
    elif kind == "count":
        torch.cuda.device_count = lambda: 2
    elif kind == "cap":
        torch.cuda.memory_reserved = lambda index: guard.MAX_OWN_RESERVED + 1
    else:
        env["CUDA_VISIBLE_DEVICES"] = "1"
    with pytest.raises(guard.TC2AccessError):
        guard.check_running_resources(torch, identity, run, env=env)


@pytest.mark.parametrize("metadata", ["NumCPUs=11", "NumNodes=2", "AllocTRES=cpu=2,mem=31G,gres/gpu=1",
                                      "BatchHost=TC2N02", "TimeLimit=00:00:00"])
def test_scheduler_resource_and_process_consistency(metadata):
    record = ("JobId=42050 JobState=RUNNING Partition=MGPU-TC2 QOS=normal NodeList=TC2N03 "
              "TresPerNode=gres:gpu:1 TimeLimit=00:15:00 " + metadata)
    with pytest.raises(guard.TC2AccessError):
        guard.slurm_allocation(valid_env(), runner(record))


def test_head_node_process_is_rejected(monkeypatch):
    monkeypatch.setattr(guard.socket, "gethostname", lambda: "TC2-HEAD")
    with pytest.raises(guard.TC2AccessError):
        guard.slurm_allocation(valid_env(), runner())


def test_subprocess_failure_is_not_bypassed():
    def failure(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    with pytest.raises(subprocess.CalledProcessError):
        guard.slurm_allocation(valid_env(), failure)

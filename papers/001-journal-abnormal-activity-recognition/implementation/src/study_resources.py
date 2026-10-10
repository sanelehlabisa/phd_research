"""Common-input throughput/memory checks, separate from predictive evidence."""

import gc
from time import perf_counter

import torch


@torch.no_grad()
def inference_measurement(model, config, device, batch_size=1, repeats=10):
    """Model-only latency; no decoding, transfer, or test sample access."""
    model.eval()
    shape = (batch_size, config.sequence_length, 3, config.height, config.width)
    inputs = torch.zeros(shape, device=device)
    cuda = device.type == "cuda"
    for _ in range(3):
        model(inputs)
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    for _ in range(repeats):
        model(inputs)
    if cuda:
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    return {
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if cuda else "CPU",
        "torch_version": str(torch.__version__),
        "dtype": str(inputs.dtype),
        "batch_size": batch_size,
        "input_shape": list(shape),
        "warmup_iterations": 3,
        "measured_iterations": repeats,
        "latency_ms_per_batch": elapsed * 1000 / repeats,
        "samples_per_second": repeats * batch_size / elapsed,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if cuda else None
        ),
        "timing_scope": "model-only forward; input already on device; no decode/transfer",
        "memory_scope": "CUDA allocated peak including model/input; CPU memory unavailable",
    }


def capacity_preflight(
    config, widths, device=None, max_depth=4, native_comparison=False
):
    """Bound the declared maximum stack and both reference families."""
    from .experiments import build_registered_model
    from .experiment_config import ExperimentConfig

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if native_comparison and device.type == "cuda":
        report = select_capacity_batch(
            config.batch_size,
            lambda spec: training_probe(spec, config, max(widths), max_depth, device),
        )
        return {**report, "hardware": torch.cuda.get_device_name(device)}
    if device.type != "cuda":
        return {
            "selected_batch_size": config.batch_size,
            **(
                {"comparison_batch_size": config.batch_size}
                if native_comparison
                else {}
            ),
            "status": "cpu_only_gpu_safety_unverified",
            "estimates": None,
            "results": [],
            "note": "CPU synthetic verification is not a Colab safety or runtime estimate.",
        }
    trials = []
    from .model import CustomConvLSTM

    selected = None
    for batch in sorted({1, 2, 4, 8, 16, 32, config.batch_size}, reverse=True):
        if batch > config.batch_size:
            continue
        safe = True
        for name in ("custom_bound", "r3d_18", "swin3d_t"):
            model = optimizer = inputs = labels = loss = None
            try:
                torch.cuda.empty_cache()
                # Check temporal/spatial upper bounds together conservatively.
                bound = ExperimentConfig.from_mapping(
                    {
                        **config.to_dict(),
                        "batch_size": batch,
                        "sequence_length": 16,
                        "height": 64,
                        "width": 64,
                    }
                )
                model = (
                    CustomConvLSTM(11, layers=[(max(widths), (3, 3))] * max_depth)
                    if name == "custom_bound"
                    else build_registered_model(name, 11, (3, 64, 64), 16, bound)
                ).to(device)
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                inputs = torch.zeros((batch, 16, 3, 64, 64), device=device)
                labels = torch.zeros(batch, dtype=torch.long, device=device)
                # One warmup allocates optimizer state; two measured full steps.
                for _ in range(1):
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = perf_counter()
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize(device)
                seconds = (perf_counter() - started) / 2
                free, total = torch.cuda.mem_get_info(device)
                peak = torch.cuda.max_memory_allocated(device)
                fits = peak <= 0.8 * total and free >= max(1024**3, 0.1 * total)
                trials.append(
                    {
                        "model": name,
                        "batch_size": batch,
                        "memory_safe": fits,
                        "peak_cuda_memory_bytes": peak,
                        "total_cuda_memory_bytes": total,
                        "training_step_seconds": seconds,
                        "samples_per_second": batch / seconds,
                    }
                )
                safe &= fits
            except torch.cuda.OutOfMemoryError:
                safe = False
                trials.append(
                    {
                        "model": name,
                        "batch_size": batch,
                        "memory_safe": False,
                        "reason": "CUDA OOM",
                    }
                )
            finally:
                del loss, labels, inputs, optimizer, model
                gc.collect()
                torch.cuda.empty_cache()
            if not safe:
                break
        if safe:
            selected = batch
            break
    report = {
        "selected_batch_size": selected,
        "status": "complete" if selected else "failed",
        "hardware": torch.cuda.get_device_name(device),
        "results": trials,
        "scope": f"synthetic 16-frame 64x64 training; {max_depth}-layer maximum width plus R3D-18/Swin3D-T",
        "note": "Throughput only; discarded weights, no model-selection evidence. Epoch estimates exclude decoding/validation/ZIP.",
    }
    return report


def select_capacity_batch(requested_batch, probe):
    """075: freeze separate common search/final batches; never skip a model."""
    trials = []
    report = {
        "selected_batch_size": None,
        "comparison_batch_size": None,
        "status": "failed",
        "results": trials,
        "scope": "Full Adam steps including optimizer state: search 8f/64px; final 16f/64px",
        "headroom": "peak <= 80% total and free >= max(1 GiB, 10% total)",
        "note": "Synthetic discarded weights; not predictive evidence or a runtime guarantee.",
    }
    bounds = ("custom_bound", "custom_depth5")
    for phase, frames, models, field in (
        ("search", 8, (*bounds, "r3d_18", "swin3d_t"), "selected_batch_size"),
        (
            "final_input",
            16,
            (
                *bounds,
                "paper_convlstm_adapted",
                "r3d_18",
                "mc3_18",
                "swin3d_t",
                "swin3d_s",
            ),
            "comparison_batch_size",
        ),
    ):
        for batch in (32, 16, 8, 4, 2, 1):
            if batch > requested_batch:
                continue
            safe = True
            for name in models:
                spec = dict(
                    model=name, batch_size=batch, frames=frames, size=64, phase=phase
                )
                try:
                    result = {**spec, **probe(spec)}
                except torch.cuda.OutOfMemoryError:
                    result = {**spec, "memory_safe": False, "reason": "CUDA OOM"}
                trials.append(result)
                if not result["memory_safe"]:
                    safe = False
                    break
            if safe:
                report[field] = batch
                break
        if report[field] is None:
            # A search batch alone cannot authorize the campaign.
            report["selected_batch_size"] = None
            return report
    report["status"] = "complete"
    return report


def training_probe(spec, config, width, depth, device):
    """Full optimizer-state memory measurement, always releasing failed probes."""
    from .experiments import build_registered_model
    from .experiment_config import ExperimentConfig
    from .model import CustomConvLSTM

    model = optimizer = inputs = labels = loss = None
    batch, frames, size = spec["batch_size"], spec["frames"], spec["size"]
    try:
        torch.cuda.empty_cache()
        bound = ExperimentConfig.from_mapping(
            {
                **config.to_dict(),
                "batch_size": batch,
                "sequence_length": frames,
                "height": size,
                "width": size,
            }
        )
        model = (
            CustomConvLSTM(
                11,
                layers=[(64 if spec["model"] == "custom_depth5" else width, (3, 3))]
                * (5 if spec["model"] == "custom_depth5" else depth),
            )
            if spec["model"] in {"custom_bound", "custom_depth5"}
            else build_registered_model(
                spec["model"], 11, (3, size, size), frames, bound
            )
        ).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        inputs = torch.zeros((batch, frames, 3, size, size), device=device)
        labels = torch.zeros(batch, dtype=torch.long, device=device)
        for step in range(3):
            if step == 1:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(inputs), labels)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        seconds = (perf_counter() - started) / 2
        free, total = torch.cuda.mem_get_info(device)
        peak = torch.cuda.max_memory_allocated(device)
        return dict(
            memory_safe=peak <= 0.8 * total and free >= max(1024**3, 0.1 * total),
            peak_cuda_memory_bytes=peak,
            total_cuda_memory_bytes=total,
            training_step_seconds=seconds,
            samples_per_second=batch / seconds,
        )
    finally:
        del loss, labels, inputs, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()

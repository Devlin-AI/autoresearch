import platform
import torch


def detect_hardware():
    info = {}

    info["PyTorch version"] = torch.__version__
    info["CUDA available"] = torch.cuda.is_available()

    if torch.cuda.is_available():
        info["CUDA version"] = torch.version.cuda
        info["GPU count"] = torch.cuda.device_count()

        props = torch.cuda.get_device_properties(0)
        info["GPU name"] = props.name
        info["Compute capability"] = f"sm_{props.major}{props.minor}"
        info["VRAM (GB)"] = round(props.total_memory / 1024**3, 2)
        info["Multi-processor count"] = props.multi_processor_count
        info["bfloat16 supported"] = torch.cuda.is_bf16_supported()
    else:
        info["GPU name"] = "No CUDA GPU"

    try:
        from flash_attn import flash_attn_func
        info["Flash Attention"] = "Available"
    except ImportError:
        info["Flash Attention"] = "Not available"

    info["CPU"] = platform.processor()

    try:
        import psutil
        info["RAM (GB)"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        info["RAM (GB)"] = "N/A (psutil not installed)"

    print("=" * 50)
    print("HARDWARE DETECTION")
    print("=" * 50)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("=" * 50)

    return info


def get_peak_bf16_tflops(compute_capability):
    sm_map = {
        (8, 6): 239.0,
        (8, 7): 239.0,
        (8, 9): 661.5,
        (9, 0): 989.5,
    }
    sm_num = int(compute_capability.split("_")[1])
    major = sm_num // 10
    minor = sm_num % 10
    key = (major, minor)
    return sm_map.get(key, 100.0)


if __name__ == "__main__":
    hw = detect_hardware()
    if "Compute capability" in hw:
        peak = get_peak_bf16_tflops(hw["Compute capability"])
        print(f"Peak BF16 TFLOPS: {peak}")
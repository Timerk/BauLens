# Training with the Radeon RX 6800 on Windows

Tested on DESKTOP-JRMR9BE with Windows 11 Home build 26200, an AMD Ryzen 7 9800X3D
(8 cores, 16 threads), 32 GB installed RAM, and a Radeon RX 6800 (gfx1030, 16 GB VRAM,
driver 32.0.21045.16010).

AMD's Windows wheels for gfx1030 ran both crack backbones with ROCm 10.0.0, PyTorch
2.13.0+rocm10.0.0 (HIP runtime reported by PyTorch as 7.15.26333), and torchvision
0.28.0+rocm10.0.0. The CPU environment uses PyTorch 2.15.0+cpu and ONNX Runtime 1.32.0.
Both environments use CPython 3.12.14. The ROCm wheels came from
[AMD's PyTorch installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html?rocm-ver=10.0.0).

## Set up the environments

Run from the checkout in PowerShell. Exporting the locked requirements supplies the same
project dependencies apart from PyTorch and torchvision. Use the explicit ROCm executable
for training and evaluation. Use the locked CPU environment for export and benchmarking.

```powershell
uv python install 3.12
uv sync --locked --python 3.12
uv run --locked baulens data download all

uv venv .venv-rocm --python 3.12
$rocmPython = '.venv-rocm/Scripts/python.exe'
uv pip install --python $rocmPython --index https://stable.repo.amd.com/rocm/whl-next/ --index-strategy unsafe-best-match 'torch[device-gfx1030]==2.13.0+rocm10.0.0' 'torchvision[device-gfx1030]==0.28.0+rocm10.0.0'
uv export --locked --no-dev --no-emit-project --no-emit-package torch --no-emit-package torchvision --no-hashes --no-header --output-file data/raw/rocm-project-requirements.txt
uv pip install --python $rocmPython -r data/raw/rocm-project-requirements.txt
uv pip install --python $rocmPython --no-deps --no-sources -e .
& $rocmPython -c "import torch; print([(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())])"
$env:HIP_VISIBLE_DEVICES = '0'  # RX 6800 on this machine. Check the listing first.
```

## Verify training

Save the probe below to `artifacts/rocm_probe.py` and run
`& $rocmPython -u artifacts/rocm_probe.py`. The tensor multiplication returned
`[[7, 10], [15, 22]]`. The first synchronized training step took 21.16 s for
ResNet18 and 14.10 s for EfficientNet-B0. The next three steps took 34-45 ms and
70-71 ms, respectively, with a preloaded synthetic batch on the GPU. MIOpen reported
a missing optional CK grouped-convolution library for gfx1030, and the compiler warned
about XNACK capability. Both backbones completed the probe. No WSL2 install or DirectML
install was needed.

```python
import time

import torch

from baulens.cracks.model import build_model, configure_trainable, set_train_mode
from baulens.seed import seed_everything

device = torch.device("cuda")
assert "RX 6800" in torch.cuda.get_device_name(device)
seed_everything(42)
torch.use_deterministic_algorithms(True, warn_only=True)
a = torch.arange(1, 5, device=device, dtype=torch.float32).reshape(2, 2)
print((a @ a).cpu())

for architecture in ("resnet18", "efficientnet_b0"):
    model = build_model(architecture, pretrained=False).to(device)
    stages = configure_trainable(model, architecture, 2)
    set_train_mode(model, stages)
    opt = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
    x = torch.randn(64, 3, 224, 224, device=device)
    y = torch.randint(0, 2, (64, 1), device=device).float()
    for step in range(4):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize(device)
        print(architecture, step, time.perf_counter() - start, loss.item())
```

## Run the crack experiments

Use `configs/cracks_full.yaml` for ResNet18 and `configs/cracks_efficientnet.yaml`
for EfficientNet-B0. The full configs save a checkpoint at each completed epoch.
Run the same train command after an interruption to resume. It uses the same shuffle
and augmentation seeds per epoch on the same device and software versions. Changing
the config, data, or split requires a fresh checkpoint directory. Checkpoints and model
bundles are under the configured `artifacts_dir`.

```powershell
$config = 'configs/cracks_full.yaml'
.venv-rocm/Scripts/baulens.exe cracks train --config $config
.venv-rocm/Scripts/baulens.exe cracks evaluate --config $config
uv run --locked baulens cracks export --config $config
uv run --locked baulens cracks benchmark --config $config
```

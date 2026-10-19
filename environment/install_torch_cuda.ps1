<#
.SYNOPSIS
  Install (or verify) the CUDA build of PyTorch for BSHI training. Used on the desktop and by the laptop worker kit.

.DESCRIPTION
  1. nvidia-smi: reads the driver version (stops with a clear message if there is no NVIDIA driver).
  2. If the installed torch is not a CUDA build of the pinned version: uninstalls torch/torchvision and installs torch 2.14.1 + torchvision 0.29.1 from the cu126 index
     (works with older drivers); if that fails, retries with cu130 (needs a newer driver, about 580 or later).
  3. Runs verify_torch_cuda.py (device, free VRAM, bf16/fp32 matmul TFLOPs, channels_last conv, torch.compile status).
  The script only changes the Python environment given by -Python. It does not touch the firewall, environment variables, or Windows features.

.PARAMETER Python    Path to python.exe of the environment (default: .venv\Scripts\python.exe next to this script's parent folder).
.PARAMETER VerifyOnly Do not install anything; only check nvidia-smi and run the verification.
.PARAMETER DryRun     Print what would be done.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File environment\install_torch_cuda.ps1
  powershell -ExecutionPolicy Bypass -File environment\install_torch_cuda.ps1 -VerifyOnly
#>
param(
  [string]$Python = (Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\Scripts\python.exe"),
  [switch]$VerifyOnly,
  [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$TorchVersion = "2.14.1"
$VisionVersion = "0.29.1"
$Verify = Join-Path $PSScriptRoot "verify_torch_cuda.py"

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

Step "NVIDIA driver"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) { throw "nvidia-smi not found: install the NVIDIA driver first (https://www.nvidia.com/Download/index.aspx), then rerun." }
$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader).Trim()
Write-Host "GPU: $gpu"
$driver = [double](($gpu -split ",")[1].Trim())
if ($driver -lt 527) { Write-Warning "Driver $driver is older than 527: CUDA 12.x wheels will not run. Update the driver." }

if (-not (Test-Path $Python)) { throw "Python not found at $Python (pass -Python path\to\python.exe)." }
Step "Python"
& $Python --version

function Installed-Torch {
  $v = & $Python -c "import importlib.metadata as m; print(m.version('torch'))" 2>$null
  if ($LASTEXITCODE -ne 0) { return "" }
  return ($v | Select-Object -First 1).Trim()
}

if (-not $VerifyOnly) {
  $have = Installed-Torch
  Write-Host "installed torch: '$have'"
  if ($have -like "$TorchVersion+cu*") {
    Write-Host "CUDA build of the pinned version is already installed; skipping the install."
  } else {
    $indexes = @("cu126", "cu130")
    $ok = $false
    foreach ($tag in $indexes) {
      Step "Install torch $TorchVersion + torchvision $VisionVersion from the $tag index"
      $url = "https://download.pytorch.org/whl/$tag"
      if ($DryRun) { Write-Host "[dry run] pip uninstall -y torch torchvision ; pip install --no-deps torch==$TorchVersion torchvision==$VisionVersion --index-url $url"; $ok = $true; break }
      & $Python -m pip uninstall -y torch torchvision
      & $Python -m pip install --no-deps "torch==$TorchVersion" "torchvision==$VisionVersion" --index-url $url
      if ($LASTEXITCODE -eq 0) { $ok = $true; break }
      Write-Warning "The $tag install failed (exit $LASTEXITCODE)."
    }
    if (-not $ok) { throw "Neither cu126 nor cu130 could be installed. Send the output above back." }
  }
}

if ($DryRun) { Write-Host "[dry run] would run $Verify"; return }
Step "Verify"
& $Python $Verify
if ($LASTEXITCODE -ne 0) { throw "CUDA is not usable with this torch build (see the JSON above)." }
Write-Host "`nCUDA PyTorch OK" -ForegroundColor Green

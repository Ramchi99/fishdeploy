# Run this in PowerShell (Admin) before starting a vision session.
# Finds the USB camera, binds it (if needed), and attaches it to WSL.
#
# Usage: Right-click → "Run with PowerShell" (as Admin)
#        Or from Admin PowerShell: .\scripts\attach_camera_wsl.ps1

$ErrorActionPreference = "Stop"

# Find first USB device matching common camera keywords
$line = usbipd list | Select-String -Pattern "Video|Camera|Webcam|UVC" | Select-Object -First 1

if (-not $line) {
    Write-Host "ERROR: No USB camera found. Is it plugged in?" -ForegroundColor Red
    exit 1
}

$busid = ($line.Line.Trim() -split '\s+')[0]
Write-Host "Found camera at BUSID: $busid  ($($line.Line.Trim()))" -ForegroundColor Cyan

# Bind (silently skip if already bound)
usbipd bind --busid $busid 2>$null
Write-Host "Bound." -ForegroundColor Green

# Attach to WSL
usbipd attach --wsl --busid $busid
Write-Host "Camera attached to WSL. Run 'ls /dev/video*' in WSL to confirm." -ForegroundColor Green

# CIFAR-10 Autoencoder - CPU Build Script for Windows
# For Intel Core i5 and other systems without NVIDIA GPU
# PowerShell 5.1 compatible

param(
    [switch]$Clean,
    [switch]$Rebuild,
    [switch]$Release,
    [switch]$Debug,
    [string]$Generator = "Visual Studio 17 2022"
)

# Color output functions
function Write-Success { param([string]$msg) Write-Host $msg -ForegroundColor Green }
function Write-Info { param([string]$msg) Write-Host $msg -ForegroundColor Cyan }
function Write-Warning { param([string]$msg) Write-Host $msg -ForegroundColor Yellow }
function Write-Error { param([string]$msg) Write-Host $msg -ForegroundColor Red }

# Header
Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "   CIFAR-10 Autoencoder - CPU Build" -ForegroundColor Magenta
Write-Host "   Windows (Intel Core i5 Compatible)" -ForegroundColor Magenta
Write-Host "========================================`n" -ForegroundColor Magenta

# Get script directory and project root
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR
Set-Location $PROJECT_ROOT

Write-Info "Project Root: $PROJECT_ROOT"

# Determine build configuration
$BUILD_CONFIG = "Release"
if ($Debug) {
    $BUILD_CONFIG = "Debug"
    Write-Info "Build Configuration: Debug"
} else {
    Write-Info "Build Configuration: Release"
}

# Build directory
$BUILD_DIR = Join-Path $PROJECT_ROOT "build"

# Clean build if requested
if ($Clean -or $Rebuild) {
    Write-Warning "`nCleaning existing build directory..."
    if (Test-Path $BUILD_DIR) {
        Remove-Item -Recurse -Force $BUILD_DIR
        Write-Success "Build directory cleaned."
    }
}

# Create build directory
if (-not (Test-Path $BUILD_DIR)) {
    Write-Info "`nCreating build directory..."
    New-Item -ItemType Directory -Path $BUILD_DIR | Out-Null
    Write-Success "Build directory created: $BUILD_DIR"
}

# Check for CMake
Write-Info "`nChecking for CMake..."
try {
    $cmakeVersion = cmake --version 2>&1 | Select-Object -First 1
    Write-Success "Found: $cmakeVersion"
} catch {
    Write-Error "CMake not found! Please install CMake 3.18 or later."
    Write-Info "Download from: https://cmake.org/download/"
    Write-Info "Or run: .\scripts\setup_windows.ps1"
    exit 1
}

# Check for Visual Studio
Write-Info "`nChecking for Visual Studio..."
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -property installationPath
    if ($vsPath) {
        Write-Success "Found Visual Studio: $vsPath"
    } else {
        Write-Warning "Visual Studio not found via vswhere"
    }
} else {
    Write-Warning "vswhere.exe not found - Visual Studio may not be installed"
    Write-Info "Download from: https://visualstudio.microsoft.com/downloads/"
}

# Configure CMake (CPU-only, no CUDA)
Write-Info "`n========================================" 
Write-Info "Configuring CMake (CPU-only build)..."
Write-Info "========================================" 

Set-Location $BUILD_DIR

$cmakeArgs = @(
    "-G", $Generator,
    "-A", "x64",
    "-DENABLE_CUDA=OFF",  # Explicitly disable CUDA
    ".."
)

Write-Info "CMake Command: cmake $($cmakeArgs -join ' ')"
Write-Info ""

$cmakeConfig = cmake @cmakeArgs 2>&1
$cmakeExitCode = $LASTEXITCODE

# Display CMake output
$cmakeConfig | ForEach-Object {
    $line = $_.ToString()
    if ($line -match "error|ERROR") {
        Write-Error $line
    } elseif ($line -match "warning|WARNING") {
        Write-Warning $line
    } elseif ($line -match "Building CPU-only version") {
        Write-Success $line
    } else {
        Write-Host $line
    }
}

if ($cmakeExitCode -ne 0) {
    Write-Error "`nCMake configuration failed!"
    Set-Location $PROJECT_ROOT
    exit 1
}

Write-Success "`nCMake configuration completed successfully!"

# Build the project
Write-Info "`n========================================" 
Write-Info "Building project..."
Write-Info "========================================`n" 

$buildArgs = @(
    "--build", ".",
    "--config", $BUILD_CONFIG,
    "--parallel"
)

Write-Info "Build Command: cmake $($buildArgs -join ' ')"
Write-Info ""

$buildOutput = cmake @buildArgs 2>&1
$buildExitCode = $LASTEXITCODE

# Display build output
$buildOutput | ForEach-Object {
    $line = $_.ToString()
    if ($line -match "error|ERROR|: error") {
        Write-Error $line
    } elseif ($line -match "warning|WARNING") {
        Write-Warning $line
    } elseif ($line -match "Build succeeded|succeeded") {
        Write-Success $line
    } else {
        Write-Host $line
    }
}

Set-Location $PROJECT_ROOT

if ($buildExitCode -ne 0) {
    Write-Error "`nBuild failed!"
    exit 1
}

# Check for executable
$exePath = Join-Path $BUILD_DIR "bin\$BUILD_CONFIG\train_autoencoder.exe"
if (-not (Test-Path $exePath)) {
    # Try alternate location
    $exePath = Join-Path $BUILD_DIR "$BUILD_CONFIG\train_autoencoder.exe"
}

if (Test-Path $exePath) {
    Write-Success "`n========================================" 
    Write-Success "   BUILD SUCCESSFUL!" 
    Write-Success "========================================" 
    Write-Info "`nExecutable created: $exePath"
    
    # Get file size
    $fileSize = (Get-Item $exePath).Length
    $fileSizeKB = [math]::Round($fileSize / 1KB, 2)
    Write-Info "File size: $fileSizeKB KB"
    
    Write-Info "`nNext steps:"
    Write-Host "  1. Download CIFAR-10 dataset:" -ForegroundColor White
    Write-Host "     .\scripts\download_cifar10.ps1" -ForegroundColor Yellow
    Write-Host "  2. Run training:" -ForegroundColor White
    Write-Host "     .$exePath" -ForegroundColor Yellow
    Write-Info "`nNote: This is a CPU-only build. For GPU support (Phases 2-4), use Google Colab."
} else {
    Write-Error "`nExecutable not found after successful build!"
    Write-Info "Expected location: $exePath"
    Write-Info "Please check build output for errors."
    exit 1
}

Write-Host ""

# Download CIFAR-10 Dataset for Windows
# This script downloads and extracts the CIFAR-10 binary dataset

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CIFAR-10 Dataset Download" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Configuration
$dataDir = "data"
$cifar10Dir = Join-Path $dataDir "cifar-10-batches-bin"
$downloadUrl = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
$downloadFile = Join-Path $dataDir "cifar-10-binary.tar.gz"

# Create data directory if it doesn't exist
if (-not (Test-Path $dataDir)) {
    Write-Host "Creating data directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
}

# Check if already downloaded
if (Test-Path $cifar10Dir) {
    Write-Host "CIFAR-10 dataset already exists in $cifar10Dir" -ForegroundColor Green
    $response = Read-Host "Do you want to re-download? (y/n)"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Skipping download." -ForegroundColor Yellow
        exit 0
    }
    Remove-Item -Recurse -Force $cifar10Dir
}

# Download CIFAR-10
Write-Host "`nDownloading CIFAR-10 dataset (~170 MB)..." -ForegroundColor Yellow
Write-Host "URL: $downloadUrl" -ForegroundColor Gray

try {
    # Use Invoke-WebRequest with progress
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $downloadUrl -OutFile $downloadFile -UseBasicParsing
    $ProgressPreference = 'Continue'
    
    Write-Host "Download complete!" -ForegroundColor Green
    
    # Get file size
    $fileSize = (Get-Item $downloadFile).Length / 1MB
    Write-Host "Downloaded: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Gray
    
} catch {
    Write-Host "Error downloading dataset: $_" -ForegroundColor Red
    exit 1
}

# Extract archive
Write-Host "`nExtracting archive..." -ForegroundColor Yellow

try {
    # Check if tar is available (Windows 10 1803+)
    $tarExists = Get-Command tar -ErrorAction SilentlyContinue
    
    if ($tarExists) {
        Write-Host "Using Windows built-in tar..." -ForegroundColor Gray
        tar -xzf $downloadFile -C $dataDir
    } else {
        # Fallback: Use 7-Zip if available
        $7zipPath = "C:\Program Files\7-Zip\7z.exe"
        if (Test-Path $7zipPath) {
            Write-Host "Using 7-Zip..." -ForegroundColor Gray
            & $7zipPath x $downloadFile -o"$dataDir" -y | Out-Null
            # Extract inner tar file
            $tarFile = Join-Path $dataDir "cifar-10-binary.tar"
            if (Test-Path $tarFile) {
                & $7zipPath x $tarFile -o"$dataDir" -y | Out-Null
                Remove-Item $tarFile
            }
        } else {
            Write-Host "ERROR: Neither tar nor 7-Zip found!" -ForegroundColor Red
            Write-Host "Please install 7-Zip or use Windows 10 1803+ with built-in tar" -ForegroundColor Yellow
            Write-Host "Download 7-Zip: https://www.7-zip.org/download.html" -ForegroundColor Yellow
            exit 1
        }
    }
    
    Write-Host "Extraction complete!" -ForegroundColor Green
    
} catch {
    Write-Host "Error extracting archive: $_" -ForegroundColor Red
    exit 1
}

# Clean up downloaded archive
Write-Host "`nCleaning up..." -ForegroundColor Yellow
Remove-Item $downloadFile -Force
Write-Host "Removed temporary archive file" -ForegroundColor Gray

# Verify extraction
Write-Host "`nVerifying dataset..." -ForegroundColor Yellow

$expectedFiles = @(
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
    "test_batch.bin",
    "batches.meta.txt"
)

$allFilesExist = $true
foreach ($file in $expectedFiles) {
    $filePath = Join-Path $cifar10Dir $file
    if (-not (Test-Path $filePath)) {
        Write-Host "  Missing: $file" -ForegroundColor Red
        $allFilesExist = $false
    } else {
        Write-Host "  Found: $file" -ForegroundColor Green
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
if ($allFilesExist) {
    Write-Host "✓ CIFAR-10 dataset ready!" -ForegroundColor Green
    Write-Host "Location: $cifar10Dir" -ForegroundColor Gray
    Write-Host "`nDataset contains:" -ForegroundColor Cyan
    Write-Host "  - 50,000 training images (5 batches)" -ForegroundColor Gray
    Write-Host "  - 10,000 test images (1 batch)" -ForegroundColor Gray
    Write-Host "  - 10 classes (airplane, automobile, bird, etc.)" -ForegroundColor Gray
} else {
    Write-Host "⚠ Some files are missing!" -ForegroundColor Red
    Write-Host "Please try downloading again or check the archive." -ForegroundColor Yellow
    exit 1
}

Write-Host "========================================`n" -ForegroundColor Cyan

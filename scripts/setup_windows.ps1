# CIFAR-10 Autoencoder - Windows Setup Script
# Checks and installs prerequisites for Windows 11
# PowerShell 5.1 compatible

Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "CIFAR-10 Autoencoder - Setup Check" -ForegroundColor Magenta
Write-Host "Windows 11 Prerequisites" -ForegroundColor Magenta
Write-Host "========================================`n" -ForegroundColor Magenta

$missingTools = @()
$warnings = @()

# Check CMake
Write-Host "[1/4] Checking CMake..." -ForegroundColor Cyan -NoNewline
try {
    $cmakeVersion = cmake --version 2>&1 | Select-Object -First 1
    if ($cmakeVersion -match "cmake version (\d+\.\d+)") {
        $version = [version]$matches[1]
        if ($version -ge [version]"3.18") {
            Write-Host " OK ($cmakeVersion)" -ForegroundColor Green
        } else {
            Write-Host " WARNING: Version $version < 3.18" -ForegroundColor Yellow
            $warnings += "CMake version too old (need 3.18+)"
        }
    }
} catch {
    Write-Host " NOT FOUND" -ForegroundColor Red
    $missingTools += "CMake"
    Write-Host "  Install: winget install Kitware.CMake" -ForegroundColor Yellow
}

# Check Visual Studio
Write-Host "[2/4] Checking Visual Studio..." -ForegroundColor Cyan -NoNewline
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -property installationPath 2>$null
    if ($vsPath) {
        $vsVersion = & $vsWhere -latest -property displayName 2>$null
        Write-Host " OK ($vsVersion)" -ForegroundColor Green
    } else {
        Write-Host " NOT FOUND" -ForegroundColor Red
        $missingTools += "Visual Studio Build Tools"
        Write-Host "  Install: winget install Microsoft.VisualStudio.2022.BuildTools" -ForegroundColor Yellow
        Write-Host "  Select Desktop development with C++ workload" -ForegroundColor Yellow
    }
} else {
    Write-Host " NOT FOUND" -ForegroundColor Red
    $missingTools += "Visual Studio Build Tools"
    Write-Host "  Install: winget install Microsoft.VisualStudio.2022.BuildTools" -ForegroundColor Yellow
    Write-Host "  Select Desktop development with C++ workload" -ForegroundColor Yellow
}

# Check Git
Write-Host "[3/4] Checking Git..." -ForegroundColor Cyan -NoNewline
try {
    $gitVersion = git --version 2>&1
    Write-Host " OK ($gitVersion)" -ForegroundColor Green
} catch {
    Write-Host " NOT FOUND" -ForegroundColor Red
    $missingTools += "Git"
    Write-Host "  Install: winget install Git.Git" -ForegroundColor Yellow
}

# Check Python
Write-Host "[4/4] Checking Python..." -ForegroundColor Cyan -NoNewline
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+\.\d+)") {
        $version = [version]$matches[1]
        if ($version -ge [version]"3.8") {
            Write-Host " OK ($pythonVersion)" -ForegroundColor Green
        } else {
            Write-Host " WARNING: Version $version < 3.8" -ForegroundColor Yellow
            $warnings += "Python version too old (need 3.8+)"
        }
    }
} catch {
    Write-Host " NOT FOUND" -ForegroundColor Red
    $missingTools += "Python"
    Write-Host "  Install: winget install Python.Python.3.11" -ForegroundColor Yellow
}

# Check GPU (informational only)
Write-Host "`n[GPU] Checking graphics hardware..." -ForegroundColor Cyan
try {
    $gpu = Get-WmiObject Win32_VideoController | Select-Object -First 1 -ExpandProperty Name
    Write-Host "  Detected: $gpu" -ForegroundColor White
    
    if ($gpu -match "NVIDIA") {
        Write-Host "  Status: NVIDIA GPU detected - can run all phases locally" -ForegroundColor Green
    } elseif ($gpu -match "Intel|AMD") {
        Write-Host "  Status: No NVIDIA GPU - use Google Colab for Phases 2-4" -ForegroundColor Yellow
        Write-Host "  Note: Phase 1 (CPU baseline) will run locally" -ForegroundColor Cyan
    }
} catch {
    Write-Host "  Could not detect GPU" -ForegroundColor Yellow
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Setup Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($missingTools.Count -eq 0 -and $warnings.Count -eq 0) {
    Write-Host "`nAll prerequisites installed!" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "1. Download dataset: .\scripts\download_cifar10.ps1" -ForegroundColor White
    Write-Host "2. Build project: .\scripts\build_cpu.ps1" -ForegroundColor White
    Write-Host "3. Run training: .\build\bin\Release\train_autoencoder.exe" -ForegroundColor White
} else {
    if ($missingTools.Count -gt 0) {
        Write-Host "`nMissing tools:" -ForegroundColor Red
        foreach ($tool in $missingTools) {
            Write-Host "  - $tool" -ForegroundColor Red
        }
    }
    
    if ($warnings.Count -gt 0) {
        Write-Host "`nWarnings:" -ForegroundColor Yellow
        foreach ($warning in $warnings) {
            Write-Host "  - $warning" -ForegroundColor Yellow
        }
    }
}

Write-Host "`n========================================`n" -ForegroundColor Cyan

# Offer to install missing tools
if ($missingTools.Count -gt 0) {
    $response = Read-Host "Install missing tools automatically? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "`nInstalling missing tools..." -ForegroundColor Cyan
        
        if ($missingTools -contains "CMake") {
            Write-Host "Installing CMake..." -ForegroundColor Yellow
            winget install Kitware.CMake --silent
        }
        
        if ($missingTools -contains "Visual Studio Build Tools") {
            Write-Host "Installing Visual Studio Build Tools..." -ForegroundColor Yellow
            Write-Host "NOTE: You must manually select Desktop development with C++ workload" -ForegroundColor Yellow
            winget install Microsoft.VisualStudio.2022.BuildTools
        }
        
        if ($missingTools -contains "Git") {
            Write-Host "Installing Git..." -ForegroundColor Yellow
            winget install Git.Git --silent
        }
        
        if ($missingTools -contains "Python") {
            Write-Host "Installing Python..." -ForegroundColor Yellow
            winget install Python.Python.3.11 --silent
        }
        
        Write-Host "`nInstallation complete!" -ForegroundColor Green
        Write-Host "Please restart PowerShell and run this script again." -ForegroundColor Yellow
    }
}

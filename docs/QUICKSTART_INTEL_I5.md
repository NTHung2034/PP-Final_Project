# Quick Start Guide - Intel Core i5 Windows 11

## ✅ Setup Complete!

Your Intel Core i5 system is now configured for CIFAR-10 Autoencoder development.

### What Was Done:

1. **Fixed CMakeLists.txt**
   - Made CUDA optional (OFF by default)
   - Added CPU-only build configuration
   - Compatible with MSVC compiler

2. **Created Windows Scripts**
   - `scripts/setup_windows.ps1` - Dependency checker
   - `scripts/build_cpu.ps1` - CPU-only build script
   - `scripts/download_cifar10.ps1` - Dataset downloader

3. **Installed Dependencies**
   - ✅ CMake 4.2.0
   - ✅ Visual Studio Build Tools 2022
   - ✅ Git 2.44.0
   - ✅ Python 3.12.2

4. **Added Memory Pool Implementation**
   - Basic CPU memory pooling in `src/utils/memory_pool.cpp`
   - Ready for Phase 1 training

### 🚀 Next Steps:

#### Step 1: Restart PowerShell
```powershell
# Close and reopen PowerShell to load new PATH
```

#### Step 2: Verify CMake Installation
```powershell
cmake --version
# Should show: cmake version 4.2.0
```

#### Step 3: Download CIFAR-10 Dataset
```powershell
.\scripts\download_cifar10.ps1
```

#### Step 4: Build CPU Version
```powershell
.\scripts\build_cpu.ps1
```

#### Step 5: Run Phase 1 Training
```powershell
.\build\bin\Release\train_autoencoder.exe
```

### 📊 Your System Configuration:

- **CPU:** Intel Core i5
- **GPU:** Intel Iris Xe Graphics (no CUDA support)
- **OS:** Windows 11 x64
- **Compiler:** MSVC (Visual Studio 2022)

### ⚠️ Important Notes:

**Phase 1 (CPU Baseline):**
- ✅ Runs locally on your Intel Core i5
- Uses OpenMP for multi-core parallelization
- Expected training time: ~2-4 hours

**Phases 2-4 (GPU Optimization):**
- ❌ Cannot run locally (requires NVIDIA GPU)
- ✅ Use Google Colab (free GPU runtime)
- See `docs/PHASE_2_GUIDE.md` for Colab setup

### 🛠️ Troubleshooting:

**If CMake is not recognized:**
```powershell
# Restart PowerShell, or manually add to PATH:
$env:Path += ";C:\Program Files\CMake\bin"
```

**If Visual Studio Build Tools not found:**
1. Open Visual Studio Installer
2. Modify Visual Studio Build Tools 2022
3. Select "Desktop development with C++"
4. Install

**If build fails:**
```powershell
# Clean rebuild
.\scripts\build_cpu.ps1 -Rebuild
```

### 📝 Development Workflow:

```powershell
# Phase 1 (Local - CPU)
cd D:\University\Parallel_programming\PP-Final_Project
.\scripts\build_cpu.ps1
.\build\bin\Release\train_autoencoder.exe

# Phases 2-4 (Google Colab - GPU)
# Upload src/gpu/ folder to Colab
# See docs/PHASE_2_GUIDE.md
```

### 📚 Documentation:

- `README.md` - Complete project overview
- `docs/PHASE_1_GUIDE.md` - CPU implementation details
- `docs/PHASE_2_GUIDE.md` - GPU porting guide (Colab)
- `docs/PHASE_3_GUIDE.md` - Memory optimization (Colab)
- `docs/PHASE_4_GUIDE.md` - Advanced optimization (Colab)

---

**Ready to start!** Close PowerShell, reopen it, and run the commands above.

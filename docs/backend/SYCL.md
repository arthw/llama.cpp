# llama.cpp for SYCL

- [Background](#background)
- [Recommended Release](#recommended-release)
- [News](#news)
- [OS](#os)
- [Hardware](#hardware)
- [Docker](#docker)
- [Linux](#linux)
- [Windows](#windows)
- [Environment Variable](#environment-variable)
- [Known Issue](#known-issues)
- [Q&A](#qa)
- [TODO](#todo)

## Background

**SYCL** is a high-level parallel programming model designed to improve developers productivity writing code across various hardware accelerators such as CPUs, GPUs, and FPGAs. It is a single-source language designed for heterogeneous computing and based on standard C++17.

**oneAPI** is an open ecosystem and a standard-based specification, supporting multiple architectures including but not limited to intel CPUs, GPUs and FPGAs. The key components of the oneAPI ecosystem include:

- **DPCPP** *(Data Parallel C++)*: The primary oneAPI SYCL implementation, which includes the icpx/icx Compilers.
- **oneAPI Libraries**: A set of highly optimized libraries targeting multiple domains *(e.g. oneMKL and oneDNN)*.
- **oneAPI LevelZero**: A high performance low level interface for fine-grained control over intel iGPUs and dGPUs.
- **Nvidia & AMD Plugins**: These are plugins extending oneAPI's DPCPP support to SYCL on Nvidia and AMD GPU targets.

### Llama.cpp + SYCL

The llama.cpp SYCL backend is designed to support **Intel GPU** firstly. Based on the cross-platform feature of SYCL, it also supports other vendor GPUs: Nvidia and AMD.

## Recommended Release

The SYCL backend would be broken by some PRs due to no online CI.

The following release is verified with good quality:

|Commit ID|Tag|Release|Verified  Platform| Update date|
|-|-|-|-|-|
|3bcd40b3c593d14261fb2abfabad3c0fb5b9e318|b4040 |[llama-b4040-bin-win-sycl-x64.zip](https://github.com/ggerganov/llama.cpp/releases/download/b4040/llama-b4040-bin-win-sycl-x64.zip) |Arc770/Linux/oneAPI 2024.1<br>MTL Arc GPU/Windows 11/oneAPI 2024.1| 2024-11-19|
|fb76ec31a9914b7761c1727303ab30380fd4f05c|b3038 |[llama-b3038-bin-win-sycl-x64.zip](https://github.com/ggerganov/llama.cpp/releases/download/b3038/llama-b3038-bin-win-sycl-x64.zip) |Arc770/Linux/oneAPI 2024.1<br>MTL Arc GPU/Windows 11/oneAPI 2024.1||

For CI and performance test summary, please refer to [llama.cpp CI for SYCL Backend](https://github.com/NeoZhangJianyu/ci_log/tree/main/llama.cpp).

## News

- 2024.11
  - Use syclcompat to improve the performance on some platforms. This requires to use oneAPI 2025.0 or newer.

- 2024.8
  - Use oneDNN as the default GEMM library, improve the compatibility for new Intel GPUs.

- 2024.6-7
  - Performance is increased: 37 -> 42.9 tokens/s of llama-2-7b.Q4_0 on Arc770.

- 2024.5
  - Performance is increased: 34 -> 37 tokens/s of llama-2-7b.Q4_0 on Arc770.
  - Arch Linux is verified successfully.

- 2024.4
  - Support data types: GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M.

- 2024.3
  - Release binary files of Windows.
  - A blog is published: **Run LLM on all Intel GPUs Using llama.cpp**: [intel.com](https://www.intel.com/content/www/us/en/developer/articles/technical/run-llm-on-all-gpus-using-llama-cpp-artical.html) or [medium.com](https://medium.com/@jianyu_neo/run-llm-on-all-intel-gpus-using-llama-cpp-fd2e2dcbd9bd).
  - New base line is ready: [tag b2437](https://github.com/ggerganov/llama.cpp/tree/b2437).
  - Support multiple cards: **--split-mode**: [none|layer]; not support [row], it's on developing.
  - Support to assign main GPU by **--main-gpu**, replace $GGML_SYCL_DEVICE.
  - Support detecting all GPUs with level-zero and same top **Max compute units**.
  - Support OPs
    - hardsigmoid
    - hardswish
    - pool2d

- 2024.1
  - Create SYCL backend for Intel GPU.
  - Support Windows build

## OS

| OS      | Status  | Verified                                       |
|---------|---------|------------------------------------------------|
| Linux   | Support | Ubuntu 22.04, Fedora Silverblue 39, Arch Linux |
| Windows | Support | Windows 11                                     |


## Hardware

### Intel GPU

SYCL backend supports Intel GPU Family:

- Intel Data Center Max Series
- Intel Flex Series, Arc Series
- Intel Built-in Arc GPU
- Intel iGPU in Core CPU (11th Generation Core CPU and newer, refer to [oneAPI supported GPU](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-base-toolkit-system-requirements.html#inpage-nav-1-1)).

#### Verified devices

| Intel GPU                     | Status  | Verified Model                        |
|-------------------------------|---------|---------------------------------------|
| Intel Data Center Max Series  | Support | Max 1550, 1100                        |
| Intel Data Center Flex Series | Support | Flex 170                              |
| Intel Arc Series              | Support | Arc 770, 730M, Arc A750               |
| Intel built-in Arc GPU        | Support | built-in Arc GPU in Meteor Lake       |
| Intel iGPU                    | Support | iGPU in 13700k, i5-1250P, i7-1260P, i7-1165G7 |

*Notes:*

- **Memory**
  - The device memory is a limitation when running a large model. The loaded model size, *`llm_load_tensors: buffer_size`*, is displayed in the log when running `./bin/llama-cli`.

  - Please make sure the GPU shared memory from the host is large enough to account for the model's size. For e.g. the *llama-2-7b.Q4_0* requires at least 8.0GB for integrated GPU and 4.0GB for discrete GPU.

- **Execution Unit (EU)**
  - If the iGPU has less than 80 EUs, the inference speed will likely be too slow for practical use.

### Other Vendor GPU

**Verified devices**

| Nvidia GPU               | Status    | Verified Model |
|--------------------------|-----------|----------------|
| Ampere Series            | Supported | A100, A4000    |
| Ampere Series *(Mobile)* | Supported | RTX 40 Series  |

| AMD GPU                  | Status       | Verified Model |
|--------------------------|--------------|----------------|
| Radeon Pro               | Experimental | W6800          |
| Radeon RX                | Experimental | 6700 XT        |

Note: AMD GPU support is highly experimental and is incompatible with F16.
Additionally, it only supports GPUs with a sub_group_size (warp size) of 32.

## Docker
The docker build option is currently limited to *intel GPU* targets.

### Build image
```sh
# Using FP16
docker build -t llama-cpp-sycl --build-arg="GGML_SYCL_F16=ON" --target light -f .devops/intel.Dockerfile .
```

*Notes*:

To build in default FP32 *(Slower than FP16 alternative)*, you can remove the `--build-arg="GGML_SYCL_F16=ON"` argument from the previous command.

You can also use the `.devops/llama-server-intel.Dockerfile`, which builds the *"server"* alternative.

### Run container

```sh
# First, find all the DRI cards
ls -la /dev/dri
# Then, pick the card that you want to use (here for e.g. /dev/dri/card1).
docker run -it --rm -v "$(pwd):/app:Z" --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card1:/dev/dri/card1 llama-cpp-sycl -m "/app/models/YOUR_MODEL_FILE" -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33
```

*Notes:*
- Docker has been tested successfully on native Linux. WSL support has not been verified yet.
- You may need to install Intel GPU driver on the **host** machine *(Please refer to the [Linux configuration](#linux) for details)*.

## Linux

### I. Setup Environment

1. **Install GPU drivers**

  - **Intel GPU**

Intel data center GPUs drivers installation guide and download page can be found here: [Get intel dGPU Drivers](https://dgpu-docs.intel.com/driver/installation.html#ubuntu-install-steps).

*Note*: for client GPUs *(iGPU & Arc A-Series)*, please refer to the [client iGPU driver installation](https://dgpu-docs.intel.com/driver/client/overview.html).

Once installed, add the user(s) to the `video` and `render` groups.

```sh
sudo usermod -aG render $USER
sudo usermod -aG video $USER
```

*Note*: logout/re-login for the changes to take effect.

Verify installation through `clinfo`:

```sh
sudo apt install clinfo
sudo clinfo -l
```

Sample output:

```sh
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Arc(TM) A770 Graphics

Platform #0: Intel(R) OpenCL HD Graphics
 `-- Device #0: Intel(R) Iris(R) Xe Graphics [0x9a49]
```

- **Nvidia GPU**

In order to target Nvidia GPUs through SYCL, please make sure the CUDA/CUBLAS native requirements *-found [here](README.md#cuda)-* are installed.

- **AMD GPU**

To target AMD GPUs with SYCL, the ROCm stack must be installed first.

2. **Install Intel® oneAPI Base toolkit**

- **For Intel GPU**

The base toolkit can be obtained from the official [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) page.

Please follow the instructions for downloading and installing the Toolkit for Linux, and preferably keep the default installation values unchanged, notably the installation path *(`/opt/intel/oneapi` by default)*.

Following guidelines/code snippets assume the default installation values. Otherwise, please make sure the necessary changes are reflected where applicable.

Upon a successful installation, SYCL is enabled for the available intel devices, along with relevant libraries such as oneAPI oneDNN for Intel GPUs.

- **Adding support to Nvidia GPUs**

**oneAPI Plugin**: In order to enable SYCL support on Nvidia GPUs, please install the [Codeplay oneAPI Plugin for Nvidia GPUs](https://developer.codeplay.com/products/oneapi/nvidia/download). User should also make sure the plugin version matches the installed base toolkit one *(previous step)* for a seamless "oneAPI on Nvidia GPU" setup.


**oneMKL for cuBlas**: The current oneMKL releases *(shipped with the oneAPI base-toolkit)* do not contain the cuBLAS backend. A build from source of the upstream [oneMKL](https://github.com/oneapi-src/oneMKL) with the *cuBLAS* backend enabled is thus required to run it on Nvidia GPUs.

```sh
git clone https://github.com/oneapi-src/oneMKL
cd oneMKL
cmake -B buildWithCublas -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DENABLE_MKLGPU_BACKEND=OFF -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_CUBLAS_BACKEND=ON -DTARGET_DOMAINS=blas
cmake --build buildWithCublas --config Release
```

- **Adding support to AMD GPUs**

**oneAPI Plugin**: In order to enable SYCL support on AMD GPUs, please install the [Codeplay oneAPI Plugin for AMD GPUs](https://developer.codeplay.com/products/oneapi/amd/download). As with Nvidia GPUs, the user should also make sure the plugin version matches the installed base toolkit.

**oneMKL for rocBlas**: The current oneMKL releases *(shipped with the oneAPI base-toolkit)* doesn't contain the rocBLAS backend. A build from source of the upstream [oneMKL](https://github.com/oneapi-src/oneMKL) with the *rocBLAS* backend enabled is thus required to run it on AMD GPUs.

```sh
git clone https://github.com/oneapi-src/oneMKL
cd oneMKL
# Find your HIPTARGET with rocminfo, under the key 'Name:'
cmake -B buildWithrocBLAS -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DENABLE_MKLGPU_BACKEND=OFF -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_ROCBLAS_BACKEND=ON -DHIPTARGETS=${HIPTARGET} -DTARGET_DOMAINS=blas
cmake --build buildWithrocBLAS --config Release
```

3. **Verify installation and environment**

In order to check the available SYCL devices on the machine, please use the `sycl-ls` command.
```sh
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

- **Intel GPU**

When targeting an intel GPU, the user should expect one or more level-zero devices among the available SYCL devices. Please make sure that at least one GPU is present, for instance [`level_zero:gpu`] in the sample output below:

```
[opencl:acc][opencl:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.10.0.17_160000]
[opencl:cpu][opencl:1] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i7-13700K OpenCL 3.0 (Build 0) [2023.16.10.0.17_160000]
[opencl:gpu][opencl:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [23.30.26918.50]
[level_zero:gpu][level_zero:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics 1.3 [1.3.26918]
```

- **Nvidia GPU**

Similarly, user targeting Nvidia GPUs should expect at least one SYCL-CUDA device [`cuda:gpu`] as below:

```
[opencl:acc][opencl:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.12.0.12_195853.xmain-hotfix]
[opencl:cpu][opencl:1] Intel(R) OpenCL, Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz OpenCL 3.0 (Build 0) [2023.16.12.0.12_195853.xmain-hotfix]
[cuda:gpu][cuda:0] NVIDIA CUDA BACKEND, NVIDIA A100-PCIE-40GB 8.0 [CUDA 12.5]
```

- **AMD GPU**

For AMD GPUs we should expect at least one SYCL-HIP device [`hip:gpu`]:

```
[opencl:cpu][opencl:0] Intel(R) OpenCL, 12th Gen Intel(R) Core(TM) i9-12900K OpenCL 3.0 (Build 0) [2024.18.6.0.02_160000]
[hip:gpu][hip:0] AMD HIP BACKEND, AMD Radeon PRO W6800 gfx1030 [HIP 60140.9]
```

### II. Build llama.cpp

#### Intel GPU

```
./examples/sycl/build.sh
```

or

```sh
# Export relevant ENV variables
source /opt/intel/oneapi/setvars.sh

# Option 1: Use FP32 (recommended for better performance in most cases)
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# Option 2: Use FP16
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON

# build all binary
cmake --build build --config Release -j -v
```

#### Nvidia GPU

```sh
# Export relevant ENV variables
export LD_LIBRARY_PATH=/path/to/oneMKL/buildWithCublas/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/path/to/oneMKL/buildWithCublas/lib:$LIBRARY_PATH
export CPLUS_INCLUDE_DIR=/path/to/oneMKL/buildWithCublas/include:$CPLUS_INCLUDE_DIR
export CPLUS_INCLUDE_DIR=/path/to/oneMKL/include:$CPLUS_INCLUDE_DIR

# Build LLAMA with Nvidia BLAS acceleration through SYCL
# Setting GGML_SYCL_DEVICE_ARCH is optional but can improve performance
GGML_SYCL_DEVICE_ARCH=sm_80 # Example architecture

# Option 1: Use FP32 (recommended for better performance in most cases)
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=NVIDIA -DGGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH} -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# Option 2: Use FP16
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=NVIDIA -DGGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH} -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON

# build all binary
cmake --build build --config Release -j -v
```

#### AMD GPU

```sh
# Export relevant ENV variables
export LD_LIBRARY_PATH=/path/to/oneMKL/buildWithrocBLAS/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/path/to/oneMKL/buildWithrocBLAS/lib:$LIBRARY_PATH
export CPLUS_INCLUDE_DIR=/path/to/oneMKL/buildWithrocBLAS/include:$CPLUS_INCLUDE_DIR

# Build LLAMA with rocBLAS acceleration through SYCL

## AMD
# Use FP32, FP16 is not supported
# Find your GGML_SYCL_DEVICE_ARCH with rocminfo, under the key 'Name:'
GGML_SYCL_DEVICE_ARCH=gfx90a # Example architecture
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=AMD -DGGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH} -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# build all binary
cmake --build build --config Release -j -v
```

### III. Run the inference

#### Retrieve and prepare model

You can refer to the general [*Prepare and Quantize*](README.md#prepare-and-quantize) guide for model prepration, or simply download [llama-2-7b.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_0.gguf) model as example.

##### Check device

1. Enable oneAPI running environment

```sh
source /opt/intel/oneapi/setvars.sh
```

2. List devices information

Similar to the native `sycl-ls`, available SYCL devices can be queried as follow:

```sh
./build/bin/llama-ls-sycl-device
```
A example of such log in a system with 1 *intel CPU* and 1 *intel GPU* can look like the following:

```
found 6 SYCL devices:
Part1:
|ID|        Device Type| Ver|                                   Name|Global mem size|
|--|-------------------|----|---------------------------------------|---------------|
| 0| [level_zero:gpu:0]| 1.3|         Intel Data Center GPU Flex 170|         16225M|
| 1| [level_zero:gpu:1]| 1.3|         Intel Data Center GPU Flex 170|         16225M|
| 2|     [opencl:gpu:0]| 3.0|         Intel Data Center GPU Flex 170|         16225M|
| 3|     [opencl:gpu:1]| 3.0|         Intel Data Center GPU Flex 170|         16225M|
| 4|     [opencl:cpu:0]| 3.0|     Intel Xeon Gold 6346 CPU @ 3.10GHz|        540700M|
| 5|     [opencl:acc:0]| 1.2|            Intel FPGA Emulation Device|        540700M|
Part2:
|ID|Max compute units|Max work group|Max subgroup|                    Driver version|
|--|-----------------|--------------|------------|----------------------------------|
| 0|              512|          1024|          32|                         1.3.27642|
| 1|              512|          1024|          32|                         1.3.27642|
| 2|              512|          1024|          32|                    23.43.27642.40|
| 3|              512|          1024|          32|                    23.43.27642.40|
| 4|               64|          8192|          64|2024.17.5.0.08_160000.xmain-hotfix|
| 5|               64|      67108864|          64|2024.17.5.0.08_160000.xmain-hotfix|

```

| Attribute              | Note                                                        |
|------------------------|-------------------------------------------------------------|
| compute capability 1.3 | Level-zero driver/runtime, recommended                      |
| compute capability 3.0 | OpenCL driver/runtime, slower than level-zero in most cases |

#### Choose level-zero devices

|Chosen Device ID|Setting|
|-|-|
|0|`export ONEAPI_DEVICE_SELECTOR="level_zero:0"` or no action|
|1|`export ONEAPI_DEVICE_SELECTOR="level_zero:1"`|
|0 & 1|`export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"`|

#### Execute

Choose one of following methods to run.

1. Script

- Use device 0:

```sh
./examples/sycl/run-llama2.sh 0
```
- Use multiple devices:

```sh
./examples/sycl/run-llama2.sh
```

2. Command line
Launch inference

There are two device selection modes:

- Single device: Use one device target specified by the user.
- Multiple devices: Automatically select the devices with the same largest Max compute-units.

| Device selection | Parameter                              |
|------------------|----------------------------------------|
| Single device    | --split-mode none --main-gpu DEVICE_ID |
| Multiple devices | --split-mode layer (default)           |

Examples:

- Use device 0:

```sh
ZES_ENABLE_SYSMAN=1 ./build/bin/llama-cli -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33 -sm none -mg 0
```

- Use multiple devices:

```sh
ZES_ENABLE_SYSMAN=1 ./build/bin/llama-cli -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33 -sm layer
```

*Notes:*

- Upon execution, verify the selected device(s) ID(s) in the output log, which can for instance be displayed as follow:

```sh
detect 1 SYCL GPUs: [0] with top Max compute units:512
```
Or
```sh
use 1 SYCL GPUs: [0] with Max compute units:512
```

## Windows

### I. Setup Environment

1. Install GPU driver

Intel GPU drivers instructions guide and download page can be found here: [Get intel GPU Drivers](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/drivers.html).

2. Install Visual Studio

If you already have a recent version of Microsoft Visual Studio, you can skip this step. Otherwise, please refer to the official download page for [Microsoft Visual Studio](https://visualstudio.microsoft.com/).

3. Install Intel® oneAPI Base toolkit

The base toolkit can be obtained from the official [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) page.

Please follow the instructions for downloading and installing the Toolkit for Windows, and preferably keep the default installation values unchanged, notably the installation path *(`C:\Program Files (x86)\Intel\oneAPI` by default)*.

Following guidelines/code snippets assume the default installation values. Otherwise, please make sure the necessary changes are reflected where applicable.

b. Enable oneAPI running environment:

- Type "oneAPI" in the search bar, then open the `Intel oneAPI command prompt for Intel 64 for Visual Studio 2022` App.

- On the command prompt, enable the runtime environment with the following:
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
```

c. Verify installation

In the oneAPI command line, run the following to print the available SYCL devices:

```
sycl-ls.exe
```

There should be one or more *level-zero* GPU devices displayed as **[ext_oneapi_level_zero:gpu]**. Below is example of such output detecting an *intel Iris Xe* GPU as a Level-zero SYCL device:

Output (example):
```
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.10.0.17_160000]
[opencl:cpu:1] Intel(R) OpenCL, 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz OpenCL 3.0 (Build 0) [2023.16.10.0.17_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Iris(R) Xe Graphics OpenCL 3.0 NEO  [31.0.101.5186]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Iris(R) Xe Graphics 1.3 [1.3.28044]
```

4. Install build tools

a. Download & install cmake for Windows: https://cmake.org/download/ (CMake can also be installed from Visual Studio Installer)
b. The new Visual Studio will install Ninja as default. (If not, please install it manually: https://ninja-build.org/)


### II. Build llama.cpp

You could download the release package for Windows directly, which including binary files and depended oneAPI dll files.

Choose one of following methods to build from source code.

1. Script

```sh
.\examples\sycl\win-build-sycl.bat
```

2. CMake

On the oneAPI command line window, step into the llama.cpp main directory and run the following:

```
@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force

# Option 1: Use FP32 (recommended for better performance in most cases)
cmake -B build -G "Ninja" -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx  -DCMAKE_BUILD_TYPE=Release

# Option 2: Or FP16
cmake -B build -G "Ninja" -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx  -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL_F16=ON

cmake --build build --config Release -j
```

Or, use CMake presets to build:

```sh
cmake --preset x64-windows-sycl-release
cmake --build build-x64-windows-sycl-release -j --target llama-cli

cmake -DGGML_SYCL_F16=ON --preset x64-windows-sycl-release
cmake --build build-x64-windows-sycl-release -j --target llama-cli

cmake --preset x64-windows-sycl-debug
cmake --build build-x64-windows-sycl-debug -j --target llama-cli
```

3. Visual Studio

You can use Visual Studio to open llama.cpp folder as a CMake project. Choose the sycl CMake presets (`x64-windows-sycl-release` or `x64-windows-sycl-debug`) before you compile the project.

*Notes:*

- In case of a minimal experimental setup, the user can build the inference executable only through `cmake --build build --config Release -j --target llama-cli`.

### III. Run the inference

#### Retrieve and prepare model

You can refer to the general [*Prepare and Quantize*](README.md#prepare-and-quantize) guide for model prepration, or simply download [llama-2-7b.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_0.gguf) model as example.

##### Check device

1. Enable oneAPI running environment

On the oneAPI command line window, run the following and step into the llama.cpp directory:
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
```

2. List devices information

Similar to the native `sycl-ls`, available SYCL devices can be queried as follow:

```
build\bin\llama-ls-sycl-device.exe
```

The output of this command in a system with 1 *intel CPU* and 1 *intel GPU* would look like the following:
```
found 6 SYCL devices:
Part1:
|ID|        Device Type| Ver|                                   Name|Global mem size|
|--|-------------------|----|---------------------------------------|---------------|
| 0| [level_zero:gpu:0]| 1.3|         Intel Data Center GPU Flex 170|         16225M|
| 1| [level_zero:gpu:1]| 1.3|         Intel Data Center GPU Flex 170|         16225M|
| 2|     [opencl:gpu:0]| 3.0|         Intel Data Center GPU Flex 170|         16225M|
| 3|     [opencl:gpu:1]| 3.0|         Intel Data Center GPU Flex 170|         16225M|
| 4|     [opencl:cpu:0]| 3.0|     Intel Xeon Gold 6346 CPU @ 3.10GHz|        540700M|
| 5|     [opencl:acc:0]| 1.2|            Intel FPGA Emulation Device|        540700M|
Part2:
|ID|Max compute units|Max work group|Max subgroup|                    Driver version|
|--|-----------------|--------------|------------|----------------------------------|
| 0|              512|          1024|          32|                         1.3.27642|
| 1|              512|          1024|          32|                         1.3.27642|
| 2|              512|          1024|          32|                    23.43.27642.40|
| 3|              512|          1024|          32|                    23.43.27642.40|
| 4|               64|          8192|          64|2024.17.5.0.08_160000.xmain-hotfix|
| 5|               64|      67108864|          64|2024.17.5.0.08_160000.xmain-hotfix|

```

| Attribute              | Note                                                      |
|------------------------|-----------------------------------------------------------|
| compute capability 1.3 | Level-zero running time, recommended                      |
| compute capability 3.0 | OpenCL running time, slower than level-zero in most cases |

#### Choose level-zero devices

|Chosen Device ID|Setting|
|-|-|
|0|`set ONEAPI_DEVICE_SELECTOR="level_zero:1"` or no action|
|1|`set ONEAPI_DEVICE_SELECTOR="level_zero:1"`|
|0 & 1|`set ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"`|

#### Execute

Choose one of following methods to run.

1. Script

```
examples\sycl\win-run-llama2.bat
```

2. Command line

Launch inference

There are two device selection modes:

- Single device: Use one device assigned by user.
- Multiple devices: Automatically choose the devices with the same biggest Max compute units.

| Device selection | Parameter                              |
|------------------|----------------------------------------|
| Single device    | --split-mode none --main-gpu DEVICE_ID |
| Multiple devices | --split-mode layer (default)           |

Examples:

- Use device 0:

```
build\bin\llama-cli.exe -m models\llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 33 -s 0 -sm none -mg 0
```

- Use multiple devices:

```
build\bin\llama-cli.exe -m models\llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 33 -s 0 -sm layer
```


Note:

- Upon execution, verify the selected device(s) ID(s) in the output log, which can for instance be displayed as follow:

```sh
detect 1 SYCL GPUs: [0] with top Max compute units:512
```
Or
```sh
use 1 SYCL GPUs: [0] with Max compute units:512
```


## Environment Variable

#### Build

| Name               | Value                                 | Function                                    |
|--------------------|---------------------------------------|---------------------------------------------|
| GGML_SYCL          | ON (mandatory)                        | Enable build with SYCL code path.<br>FP32 path - recommended for better perforemance than FP16 on quantized model|
| GGML_SYCL_TARGET   | INTEL *(default)* \| NVIDIA \| AMD    | Set the SYCL target device type.            |
| GGML_SYCL_DEVICE_ARCH | Optional (except for AMD)          | Set the SYCL device architecture, optional except for AMD. Setting the device architecture can improve the performance. See the table [--offload-arch](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/OffloadDesign.md#--offload-arch) for a list of valid architectures. |
| GGML_SYCL_F16      | OFF *(default)* \|ON *(optional)*     | Enable FP16 build with SYCL code path.      |
| CMAKE_C_COMPILER   | `icx` *(Linux)*, `icx/cl` *(Windows)* | Set `icx` compiler for SYCL code path.      |
| CMAKE_CXX_COMPILER | `icpx` *(Linux)*, `icx` *(Windows)*   | Set `icpx/icx` compiler for SYCL code path. |

#### Runtime

| Name              | Value            | Function                                                                                                                  |
|-------------------|------------------|---------------------------------------------------------------------------------------------------------------------------|
| GGML_SYCL_DEBUG   | 0 (default) or 1 | Enable log function by macro: GGML_SYCL_DEBUG                                                                             |
| ZES_ENABLE_SYSMAN | 0 (default) or 1 | Support to get free memory of GPU by sycl::aspect::ext_intel_free_memory.<br>Recommended to use when --split-mode = layer |
| GGML_SYCL_VISIBLE_DEVICES|id1,id2,...|It's like `CUDA_VISIBLE_DEVICES`, define the SYCL device ID list to visible. Like "0", "0,2", "2,1" |
| ONEAPI_DEVICE_SELECTOR|Refer to [oneapi-device-selector](https://intel.github.io/llvm-docs/EnvironmentVariables.html#oneapi-device-selector)|be used to limit the choice of devices available when the SYCL-using application is run|

##### Choose SYCL Devices in Running Time

In SYCL running time, a physical device could be mapped to two logical devices on different running times: Level-Zero and OpenCL. So it will show more devices in SYCL view. But we need avoid to run code on these two logical devices on same physical device in same time.

The SCYL backend supports dGPU or iGPU in same machine.

##### SYCL Backend Rule:

|Mode|Explain|Example|Recommend Cases|Note|
|-|-|-|-|-|
|Normal|Use all powest devices. Default mode. No special setting.<br>SYCL backend will detect and choose the **Level-Zero** devices which have top `Max compute units`.<br> ||Most cases of normal user.||
|Advanced|Allow user choose one or more SYCL devices which could be Level-Zero or OpenCL or both.<br>Set the device list by environment variable: **GGML_SYCL_VISIBLE_DEVICES**, like `CUDA_VISIBLE_DEVICES`.<br>SYCL backend will choose all devices by it.| `set/export GGML_SYCL_VISIBLE_DEVICES=1`<br>`set/export GGML_SYCL_VISIBLE_DEVICES=0,1`<br>`set/export GGML_SYCL_VISIBLE_DEVICES=2,1`|Use iGPU or both in dGPU + iGPU environment<br>Use a dGPU in mulitple dGPU environment.<br>Use one or more OpenCL devices|There is known issue of OpenCL device. WIP.|
|Developer|Allow SYCL developer choose one or more SYCL devices by environment varibale **ONEAPI_DEVICE_SELECTOR** with flexiable grammar.<br>Refer to [oneapi-device-selector](https://intel.github.io/llvm-docs/EnvironmentVariables.html#oneapi-device-selector).|`set/export ONEAPI_DEVICE_SELECTOR=level_zero:1`<br>`set/export ONEAPI_DEVICE_SELECTOR=opencl:*`<br>`set/export ONEAPI_DEVICE_SELECTOR=opencl:gpu;level_zero:gpu`<br>|Cover the Advanced mode. It will impact **Normal** and **Advanced** modes as low level principle.<br>Flexiable grammar support more complex device environments.|There is known issue of OpenCL device. WIP.|

##### Parameters of Llama.cpp

The parameters about device choose of llama.cpp works with SYCL backend rule to decide the final result. User could use one or all chosen devices by SYCL backend rule.

|Device|Values|Note|
|-|-|-|
|Single Device|`--split-mode=none` and `--main-gpu=id`|The value of `main-gpu` must be in the chosen device lists printed out during llama.cpp startup. Like:<br>`detect 2 SYCL level-zero GPUs:[0,1]`.<br>`main-gpu` should be set to `0` or `1`.|
|Multiple Device|`--split-mode=layer`|Default|


## Known Issues

- `Split-mode:[row]` is not supported.

## Q&A

- Error:  `error while loading shared libraries: libsycl.so.7: cannot open shared object file: No such file or directory`.

  - Potential cause: Unavailable oneAPI installation or not set ENV variables.
  - Solution: Install *oneAPI base toolkit* and enable its ENV through: `source /opt/intel/oneapi/setvars.sh`.

- General compiler error:

  - Remove **build** folder or try a clean-build.

- I can **not** see `[ext_oneapi_level_zero:gpu]` afer installing the GPU driver on Linux.

  Please double-check with `sudo sycl-ls`.

  If it's present in the list, please add video/render group to your user then **logout/login** or restart your system:

  ```
  sudo usermod -aG render $USER
  sudo usermod -aG video $USER
  ```
  Otherwise, please double-check the GPU driver installation steps.

- Can I report Ollama issue on Intel GPU to llama.cpp SYCL backend?

  No. We can't support Ollama issue directly, because we aren't familiar with Ollama.

  Sugguest reproducing on llama.cpp and report similar issue to llama.cpp. We will surpport it.

  It's same for other projects including llama.cpp SYCL backend.

- Meet issue: `Native API failed. Native API returns: -6 (PI_ERROR_OUT_OF_HOST_MEMORY) -6 (PI_ERROR_OUT_OF_HOST_MEMORY) -999 (UNKNOWN PI error)` or `failed to allocate SYCL0 buffer`

  Device Memory is not enough.

  |Reason|Solution|
  |-|-|
  |Default Context is too big. It leads to more memory usage.|Set `-c 8192` or smaller value.|
  |Model is big and require more memory than device's.|Choose smaller quantized model, like Q5 -> Q4;<br>Use more than one devices to load model.|

### **GitHub contribution**:
Please add the **[SYCL]** prefix/tag in issues/PRs titles to help the SYCL-team check/address them without delay.

## TODO

- NA

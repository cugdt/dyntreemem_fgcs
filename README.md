# Different GPU in-memory organizations of decision trees (complete, compact and adaptive) - time and memory peformance

## How to use it ?

**1. Download and install the CUDA Toolkit (tested with ver11.8) for your corresponding platform.**

#### Linux
- Verify the system has a CUDA-capable GPU.
```
lspci | grep -i nvidia
```

- Verify the system is running a supported version of Linux.
```
uname -m && cat /etc/*release
```

- Verify the system has gcc installed.
```
gcc --version
```
if not
```
sudo apt install build-essential
```

- Verify the system has the correct kernel headers and development packages installed.
```
uname -r
```

- Download the NVIDIA CUDA Toolkit.
  
The NVIDIA CUDA Toolkit is available at https://developer.nvidia.com/cuda-downloads.
Choose the platform you are using and download the NVIDIA CUDA Toolkit.
To calculate the MD5 checksum of the downloaded file, run the following:
```
md5sum <file>
```

- Handle conflicting installation methods.

More about system requirements and installation instructions of cuda toolkit, please refer to the Linux Installation Guide:
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/


#### Windows
...

More about system requirements and installation instructions of cuda toolkit, please refer to the Windows Installation Guide:
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

**2. compile the code**
#### Linux
#### Windows

**3. run**
#### Linux
#### Windows

## Basic class description and main options

- CudaWorker - CUDA worker is used to calculate fitness (+ dipoles) after successful mutation/crossover; it includes two kernel functions (dev_CalcPopClassDistAndDipolAtLeafs_Pre_V2b, dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b - and several their variations) that calculate the arrangements of samples, errors and dipoles; kernels are called from the CalcIndivDetailedErrAndClassDistAndDipol_V2b method (or one of its variations) where also the DT is copied to GPU and, finally, the results are received by the CPU; the in-memory DT representation is chosen in Worker.h file using ADDAPTIVE_TREE_REP and FULL_BINARY_TREE_REP defines;
- Worker - a base class for external computing resources (CUDA, SPARK, etc.) to outsource the time-demanding jobs
- IEvolutionaryAlg - evolutionary loop
- main.cpp - start file

# Different GPU in-memory organizations of decision trees (complete, compact and adaptive) - time and memory peformance
The software hab been extensively tested with CUDA 11.8 under Ubuntu 20 and GCC 9.4 as well as under Windows 10 using MS Visual Studio 2017 and 2019.

## How to use it ?
- install the NVIDIA driver
- install the CUDA toolkit
- build the solution
- run the solution
- analyse the results, eventually modify settings
<BR/>

**1. Install the NVIDIA driver**
##### Linux
- Verify the system has a CUDA-capable GPU.
```
lspci | grep -i nvidia
```

- Remove old installation
```
sudo apt-get purge nvidia-*
sudo apt-get update 
sudo apt upgrade
sudo apt-get autoremove
```

- Search for latest version of NVIDIA driver
```
apt search nvidia-driver
```

- Install and handle problems <br/>  
If you want to install a specific driver version, use this command:
```
sudo apt install nvidia-driver-<version>
```
Alternatively, if you are satisfied with the recommended version, use this command:
```
sudo ubuntu-drivers autoinstall
```

- Reboot and check for the installation
```
nvidia-smi
```

You can install the NVIDIA driver also by GUI using "Software & Updates" application windows

More about NVIDIA driver installation, please refer to the 
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

##### Windows
Windows Update automatically install and update NVIDIA Driver. <BR/><BR/>

**2. Install the CUDA Toolkit.**

##### Linux
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
Choose the platform you are using and download the NVIDIA CUDA Toolkit. It is available at https://developer.nvidia.com/cuda-downloads.


- Install and handle problems.
```
sudo apt update
sudo apt install cuda-toolkit-<version>
```

- Set up the development environment by modifying the PATH and LD_LIBRARY_PATH variables:
```
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

- Test the CUDA toolkit
```
nvcc -V
```

More about system requirements and installation instructions of CUDA toolkit, please refer to the Linux Installation Guide:
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/


#### Windows
- Download the NVIDIA CUDA Toolkit.
Choose the platform you are using and download the NVIDIA CUDA Toolkit. It is available at https://developer.nvidia.com/cuda-downloads.
- Install the CUDA toolkit.
- Windows exe CUDA Toolkit installation method automatically adds CUDA Toolkit specific environment variables.

More about system requirements and installation instructions of cuda toolkit, please refer to the Windows Installation Guide:
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
<BR/><BR/>

**3. Build the software**
##### Linux
```
make clean
make
```

##### Windows
- Create a new 'Hello World' CUDA project as a CUDA template project.
  * Go to File --> New --> Project…
  * NVIDIA -> CUDA 11.8 -> CUDA 11.8 Runtime
  * ...
- To compile this program, click on Build --> Build Solution
<BR/><BR/>

**4. Run the software**
##### Linux
```
./mlpdt
```

##### Windows
Running using the Debug configuration via Debug --> Start Without Debugging should yield the following prompt:
<BR/><BR/>

**5. Analyse the results, eventually modify settings**
The TEST version of the software brings ...

##### Windows
<BR/><BR/>


## Basic class description and main options
- CudaWorker - CUDA worker is used to calculate fitness (+ dipoles) after successful mutation/crossover; it includes two kernel functions (dev_CalcPopClassDistAndDipolAtLeafs_Pre_V2b, dev_CalcPopDetailedErrAndClassDistAndDipol_Post_V2b - and several their variations) that calculate the arrangements of samples, errors and dipoles; kernels are called from the CalcIndivDetailedErrAndClassDistAndDipol_V2b method (or one of its variations) where also the DT is copied to GPU and, finally, the results are received by the CPU; the in-memory DT representation is chosen in Worker.h file using ADDAPTIVE_TREE_REP and FULL_BINARY_TREE_REP defines;
- Worker - a base class for external computing resources (CUDA, SPARK, etc.) to outsource the time-demanding jobs
- IEvolutionaryAlg - evolutionary loop
- main.cpp - start file

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
cd src
make clean
make
```

##### Windows
- Create a new 'Hello World' CUDA project (a CUDA template project).
  * Go to 'File -> New -> Project…'
  * Choose 'NVIDIA -> CUDA 11.8 -> CUDA 11.8 Runtime'
  * Delete file "kernel.cu" from the project<BR/>
    Right-click the project name in Solution Explorer, select the file and click 'Remove'
  * Import the source code from the repository to the project<BR/>
    Right-click the project name in Solution Explorer, select 'Add -> Existing Item...', choose all source files from src folder (*.h, *.cpp, *.cuh, *.cu)  and click 'Add'.
  *  If needed, choose the correct CUDA version:<BR/>
     Right-click the project name in Solution Explorer, choose 'Build Dependencies' then 'Build Customizations...'
- To compile this program, click on 'Build -> Build Solution'
<BR/><BR/>

**4. Run the software**
##### Linux
```
./mlpdt
```

##### Windows
Running using the Debug configuration via 'Debug -> Start Without Debugging'.
<BR/><BR/>

**5. Analyse the results, eventually modify settings**
- The basic version of the software gives the times results for randomly generated decision trees for the chosen in-memory representation.<BR/>
<img src="/fig/complete_times.jpg" width="200"><BR/>
<img src="/fig/compact_times.jpg" width="200"><BR/>
<img src="/fig/adaptive_times.jpg" width="200"><BR/>

- The basic settings:
  * N_THREADS - number of threads
  * N_BLOCKS - number of blocks
  * DATASET_TRANING - traning data
  * TEST - test or evo launch
  * DS_REAL - float/double precision
  * FULL_BINARY_TREE_REP - complete/compact representation
  * ADDAPTIVE_TREE_REP - adaptive representation
  * ADAPTIVE_TREE_REP_SWITCHING_POINT - switching point in the adaptive representation
  * N_DIPOL_OBJECTS - number of dipole samples per node


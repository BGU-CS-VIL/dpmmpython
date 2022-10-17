<br>
<p align="center">
<img src="https://www.cs.bgu.ac.il/~dinari/images/clusters_low_slow.gif" alt="DPGMM SubClusters 2d example">
</p>

## DPMMSubClusters

This package is a Python wrapper for the [DPMMSubClusters.jl](https://github.com/BGU-CS-VIL/DPMMSubClusters.jl) Julia package and for the [DPMMSubClusters_GPU](https://github.com/BGU-CS-VIL/DPMMSubClusters_GPU) CUDA/C++ package.<br>

The package is useful for fitting, in a scalable way, a mixture model with an unknown number of components. We currently support either Multinomial or Gaussian components, but additional types of components can be easily added, as long as they belong to an exponential family. 



### Motivation

Working on a subset of 100K images from ImageNet, containing 79 classes, we have created embeddings using [SWAV](https://github.com/facebookresearch/swav), and reduced the dimension to 128 using PCA. We have compared our method with the popular scikit-learn [GMM](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) and [DPGMM](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html) with the following results:
<p align="center">
  
| Method                                              | Timing (sec) | NMI (higher is better) |
|-----------------------------------------------------|--------------|------------------------|
| *Scikit-learn's GMM* (using EM, and given the True K) | 2523         | 0.695                   |
| *Scikit-learn's DPGMM*                                | 6108         | 0.683                   | 
| DPMMpython (CPU Version)                              | 475           | 0.705                   | 

</p>


### Installation

If you wish to use only the CPU version, you may skip all the GPU related steps.

1. Install Julia from: https://julialang.org/downloads/platform
2. Add our DPMMSubCluster package from within a Julia terminal via Julia package manager:
```
] add DPMMSubClusters
```
3. Add our dpmmpython package in python: pip install dpmmpython
4. Add Environment Variables:
	#### On Linux:
	1. Add to the "PATH" environment variable the path to the Julia executable (e.g., in .bashrc add: export PATH =$PATH:$HOME/julia/julia-1.6.0/bin).
	#### On Windows:	
	1. Add to the "PATH" environment variable the path to the Julia executable (e.g., C:\Users\<USER>\AppData\Local\Programs\Julia\Julia-1.6.0\bin).
5. Install PyJulia from within a Python terminal:
```
	import julia;julia.install();
```
<b>GPU Steps:</b>

1. Install CUDA version 11.2 (or higher) from https://developer.nvidia.com/CUDA-downloads
2. git clone https://github.com/BGU-CS-VIL/DPMMSubClusters_GPU
3. Add Environment Variables:
	#### On Linux:
	1. Add "CUDA_VERSION" with the value of the version of your CUDA installation (e.g., 11.6).
	2. Make sure that CUDA_PATH exist. If it is missing add it with a path to CUDA (e.g., export CUDA_PATH=/usr/local/cuda-11.6/).
	3. Make sure that the relevant CUDA paths are included in $PATH and $LD_LIBRARY_PATH (e.g., export PATH=/usr/local/cuda-11.6/bin:$PATH, export LD_LIBRARY_PATH=/usr/local/cuda-
11.6/lib64:$LD_LIBRARY_PATH).
	#### On Windows:	
	1. Add "CUDA_VERSION" with the value of the version of your CUDA installation (e.g., 11.6).
	2. Make sure that CUDA_PATH exists. If it is missing add it with a path to CUDA (e.g., C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6).
4. Install cmake if necessary.

5. For Windows only (optional, used on for debugging purposes): Install OpenCV
	1. run Git Bash
	2. cd <YOUR_PATH_TO_DPMMSubClusters_GPU>/DPMMSubClusters
	3. ./installOCV.sh

### Building
For Windows for the CUDA/C++ package both of the build options below are viable. For Linux use
Option 2.
#### Option 1:
DPMMSubClusters.sln - Solution file for Visual Studio 2019
#### Option 2:
CMakeLists.txt
1. Run in the terminal:
```
cd <YOUR_PATH_TO_DPMMSubClusters_GPU>/DPMMSubClusters
mkdir build
cd build
cmake -S ../
```
2. Build:
* Windows: 
```cmake --build . --config Release --target ALL_BUILD```
* Linux: ```cmake --build . --config Release --target all```

### Post Build
Add Environment Variable:
* On Linux:</BR>
Add "DPMM_GPU_FULL_PATH_TO_PACKAGE_IN_LINUX" with the value of the path to the binary of the package DPMMSubClusters_GPU.</BR>
The path is: <YOUR_PATH_TO_DPMMSubClusters_GPU>/DPMMSubClusters/DPMMSubClusters.
* On Windows:</BR>
Add "DPMM_GPU_FULL_PATH_TO_PACKAGE_IN_WINDOWS" with the value of the path to the exe of the package DPMMSubClusters_GPU.</BR>
The path is: <YOUR_PATH_TO_DPMMSubClusters_GPU>\DPMMSubClusters\build\Release
\DPMMSubClusters.exe.

<b>End of GPU Steps</b>

## Precompiled Binaries -
[Windows](https://drive.google.com/file/d/1gQE6BWSseOEBW3xFTuahXJPIZI16uwj7/view?usp=sharing) <br>
[Linux](https://drive.google.com/file/d/1EWBqZG2jv4yH_O-BIwvDdn6gTJbF4mU4/view?usp=sharing)<br>
Both binaries were compiled with CUDA 11.2, note that you still need to have cuda and cudnn installed in order to use these.


### Usage Example:

```
from julia.api import Julia
jl = Julia(compiled_modules=False)
from dpmmpython.dpmmwrapper import DPMMPython
from dpmmpython.priors import niw
import numpy as np

data,gt = DPMMPython.generate_gaussian_data(10000, 2, 10, 100.0)
prior = niw(1,np.zeros(2),4,np.eye(2))
labels,_,results = DPMMPython.fit(data,100,prior = prior,verbose = True, gt = gt, gpu = False)
  
```
```
Iteration: 1 || Clusters count: 1 || Log posterior: -71190.14226686998 || Vi score: 1.990707323192506 || NMI score: 6.69243345834295e-16 || Iter Time:0.004499912261962891 || Total time:0.004499912261962891
Iteration: 2 || Clusters count: 1 || Log posterior: -71190.14226686998 || Vi score: 1.990707323192506 || NMI score: 6.69243345834295e-16 || Iter Time:0.0038819313049316406 || Total time:0.008381843566894531
...
Iteration: 98 || Clusters count: 9 || Log posterior: -40607.39498126549 || Vi score: 0.11887067921133423 || NMI score: 0.9692247699387838 || Iter Time:0.015907764434814453 || Total time:0.5749104022979736
Iteration: 99 || Clusters count: 9 || Log posterior: -40607.39498126549 || Vi score: 0.11887067921133423 || NMI score: 0.9692247699387838 || Iter Time:0.01072382926940918 || Total time:0.5856342315673828
Iteration: 100 || Clusters count: 9 || Log posterior: -40607.39498126549 || Vi score: 0.11887067921133423 || NMI score: 0.9692247699387838 || Iter Time:0.010260820388793945 || Total time:0.5958950519561768
```
```
predictions, probabilities = DPMMPython.predict(results[-1],data)
```

You can modify the number of processes by using `DPMMPython.add_procs(procs_count)`, note that you can only scale it upwards.

#### Additional Examples:
[Clustering](https://nbviewer.jupyter.org/github/BGU-CS-VIL/dpmmpython/blob/master/examples/clustering_example.ipynb)
<br>
[Multi-Process](https://nbviewer.jupyter.org/github/BGU-CS-VIL/dpmmpython/blob/master/examples/multi_process.ipynb)


#### Python 3.8/3.9
If you are having problems with the above Python version, please update PyJulia and PyCall to the latest versions, this should fix it.

### Misc

For any questions: dinari@post.bgu.ac.il

Contributions, feature requests, suggestion etc.. are welcomed.

If you use this code for your work, please cite the following works:

```
@inproceedings{dinari2019distributed,
  title={Distributed MCMC Inference in Dirichlet Process Mixture Models Using Julia},
  author={Dinari, Or and Yu, Angel and Freifeld, Oren and Fisher III, John W},
  booktitle={2019 19th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing (CCGRID)},
  pages={518--525},
  year={2019}
}

@article{dinari2022cpu,
  title={CPU-and GPU-based Distributed Sampling in Dirichlet Process Mixtures for Large-scale Analysis},
  author={Dinari, Or and Zamir, Raz and Fisher III, John W and Freifeld, Oren},
  journal={arXiv preprint arXiv:2204.08988},
  year={2022}
}
```

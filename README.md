<br>
<p align="center">
<img src="https://www.cs.bgu.ac.il/~dinari/images/clusters_low_slow.gif" alt="DPGMM SubClusters 2d example">
</p>

## DPMMSubClusters

This package is a Python wrapper for the [DPMMSubClusters.jl](https://github.com/BGU-CS-VIL/DPMMSubClusters.jl) Julia package.<br>

### Motivation

Working on a subset of 100K images from ImageNet, containing 79 classes, we have created embeddings using [SWAV](https://github.com/facebookresearch/swav), and reduced the dimension to 128 using PCA. We have compared our method with the popular scikit-learn [GMM](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) and [DPGMM](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html) with the following results:
<p align="center">
  
| Method                                              | Timing (sec) | NMI (higher is better) |
|-----------------------------------------------------|--------------|------------------------|
| *Scikit-learn's GMM* (using EM, and given the True K) | 2523         | 0.695                   |
| *Scikit-learn's DPGMM*                                | 6108         | 0.683                   | 
| DPMMpython                                          | 475           | 0.705                   | 

</p>


### Installation

```
pip install dpmmpython
```

If you already have Julia installed, install [PyJulia](https://github.com/JuliaPy/pyjulia) and add the package `DPMMSubClusters` to your julia installation. <p>
<p>
Make sure Julia path is configured correctly, e.g. you should be able to run julia by typing `julia` from the terminal, unless configured properly, PyJulia wont work.


**Installation Shortcut for Ubuntu distributions** <br>
If you do not have Julia installed, or wish to create a clean installation for the purpose of using this package. after installing (with pip), do the following:

```
import dpmmpython
dpmmpython.install()
```
Optional arguments are `install(julia_download_path = 'https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.0-linux-x86_64.tar.gz', julia_target_path = None)`, where the former specify the julia download file, and the latter the installation path, if the installation path is not specified, `$HOME$/julia` will be used.<br>
As the `install()` command edit your `.bashrc` path, before using the pacakge, the terminal should either be reset, or modify the current environment according to the julia path you specified (`$HOME$/julia/julia-1.4.0/bin` by default).

### Usage Example:

```
from dpmmpython.dpmmwrapper import DPMMPython
from dpmmpython.priors import niw
import numpy as np

data,gt = DPMMPython.generate_gaussian_data(10000, 2, 10, 100.0)
prior = niw(1,np.zeros(2),4,np.eye(2))
labels,_,sub_labels= DPMMPython.fit(data,100,prior = prior,verbose = True, gt = gt)
```
```
Iteration: 1 || Clusters count: 1 || Log posterior: -71190.14226686998 || Vi score: 1.990707323192506 || NMI score: 6.69243345834295e-16 || Iter Time:0.004499912261962891 || Total time:0.004499912261962891
Iteration: 2 || Clusters count: 1 || Log posterior: -71190.14226686998 || Vi score: 1.990707323192506 || NMI score: 6.69243345834295e-16 || Iter Time:0.0038819313049316406 || Total time:0.008381843566894531
...
Iteration: 98 || Clusters count: 9 || Log posterior: -40607.39498126549 || Vi score: 0.11887067921133423 || NMI score: 0.9692247699387838 || Iter Time:0.015907764434814453 || Total time:0.5749104022979736
Iteration: 99 || Clusters count: 9 || Log posterior: -40607.39498126549 || Vi score: 0.11887067921133423 || NMI score: 0.9692247699387838 || Iter Time:0.01072382926940918 || Total time:0.5856342315673828
Iteration: 100 || Clusters count: 9 || Log posterior: -40607.39498126549 || Vi score: 0.11887067921133423 || NMI score: 0.9692247699387838 || Iter Time:0.010260820388793945 || Total time:0.5958950519561768
```

You can modify the number of processes by using `DPMMPython.add_procs(procs_count)`, note that you can only scale it upwards.

#### Additional Examples:
[Clustering](https://nbviewer.jupyter.org/github/BGU-CS-VIL/dpmmpython/blob/master/examples/clustering_example.ipynb)
<br>
[Multi-Process](https://nbviewer.jupyter.org/github/BGU-CS-VIL/dpmmpython/blob/master/examples/multi_process.ipynb)


#### Python 3.8
Due to recent issue with the package used as interface between Julia and Python https://github.com/JuliaPy/pyjulia/issues/425 , there might be problems working with Python >= 3.8.

### Misc

For any questions: dinari@post.bgu.ac.il

Contributions, feature requests, suggestion etc.. are welcomed.

If you use this code for your work, please cite the following:

```
@inproceedings{dinari2019distributed,
  title={Distributed MCMC Inference in Dirichlet Process Mixture Models Using Julia},
  author={Dinari, Or and Yu, Angel and Freifeld, Oren and Fisher III, John W},
  booktitle={2019 19th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing (CCGRID)},
  pages={518--525},
  year={2019}
}
```

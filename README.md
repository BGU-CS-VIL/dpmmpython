## DPMMSubClusters

This package is a wrapper for [DPMMSubClusters.jl](https://github.com/BGU-CS-VIL/DPMMSubClusters.jl) Julia package.<br>

### Installation

```
pip install dpmmpython
```

If you already have Julia installed, install [PyJulia](https://github.com/JuliaPy/pyjulia) and add the package `DPMMSubClusters` to your julia installation. <p>

**The next part only works with Ubuntu distributions at the moment!** <br>
If you do not have Julia installed, or wish to create a clean installation for the purpose of using this package. after installing (with pip), do the following:

```
import dpmmpython
dpmmpython.install()
```
Optional arguments are `install(julia_download_path = 'https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.0-linux-x86_64.tar.gz', julia_target_path = None)`, where the former specify the julia download file, and the latter the installation path, if the installation path is not specified, `$HOME$/julia` will be used.

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



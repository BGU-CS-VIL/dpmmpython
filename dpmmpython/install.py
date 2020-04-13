import julia
import os
import sys
import wget
import tarfile
from julia.api import Julia



def get_julia_path_from_dir(base_dir):
    dir_content = os.listdir(base_dir)
    julia_path = base_dir
    for item in dir_content:
        if os.path.isdir(os.path.join(julia_path,item)):
            julia_path = os.path.join(julia_path,item)
            break

    return os.path.join(julia_path,'bin','julia'),os.path.join(julia_path,'bin')



def install(julia_download_path = 'https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.0-linux-x86_64.tar.gz', julia_target_path = None):
    '''
    :param julia_download_path: Path for the julia download, you can modify for your preferred version
    :param julia_target_path: Specify where to install Julia, if not specified will install in $HOME$/julia
    '''
    if julia_target_path == None:
        julia_target_path = os.path.join(os.path.expanduser("~"),'julia')
    if not os.path.isdir(julia_target_path):
        os.mkdir(julia_target_path)    
    download_path = os.path.join(julia_target_path,'julia_install.tar.gz')
    print("Downloading Julia:")
    wget.download(julia_download_path, download_path)
    print("\nExtracting...")
    tar = tarfile.open(download_path,"r:gz")
    tar.extractall(julia_target_path)
    _, partial_path = get_julia_path_from_dir(julia_target_path)
    os.environ["PATH"] += os.pathsep + partial_path
    os.system("echo '# added by dpmmpython' >> ~/.bashrc")
    os.system("echo 'export PATH=\""+partial_path+":$PATH\"' >> ~/.bashrc")
    print("Configuring PyJulia")    
    julia.install()
    julia.Julia(compiled_modules=False)
    print("Adding DPMMSubClusters package")  
    from julia import Pkg
    Pkg.add("DPMMSubClusters")
    print("Please exit the shell and restart, before attempting to use the package") 


if __name__ == "__main__":
    install()
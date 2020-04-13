import julia
import os
import sys
import wget
import tarfile



def get_julia_path_from_dir(base_dir):
    dir_content = os.listdir(base_dir)
    julia_path = base_dir
    for item in dir_content:
        if os.path.isdir(os.path.join(julia_path,item)):
            julia_path = os.path.join(julia_path,item)
            break
    julia_path = os.path.join(julia_path,'bin','julia')
    return julia_path



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
    julia_path = get_julia_path_from_dir(julia_target_path)
    print("Configuring PyJulia")
    julia.install(julia=julia_path)
    j = julia.Julia()
    j.eval('using Pkg')
    j.eval('Pkg.add("DPMMSubClusters")')  


if __name__ == "__main__":
    install()
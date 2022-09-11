import os
import platform
import julia
julia.install()
from julia.api import Julia
from dpmmpython.priors import niw, multinomial
from julia import DPMMSubClusters
import numpy as np
import subprocess
import json
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import normalized_mutual_info_score as nmi
from time import time
from datetime import datetime
from sklearn.decomposition import PCA
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups_vectorized
import gdown

# Set the full path for the PMMSubClusters.exe in windows
FULL_PATH_TO_PACKAGE_IN_WINDOWS = os.environ.get('DPMM_GPU_FULL_PATH_TO_PACKAGE_IN_WINDOWS')

# Set the full path for the PMMSubClusters in Linux
FULL_PATH_TO_PACKAGE_IN_LINUX = os.environ.get('DPMM_GPU_FULL_PATH_TO_PACKAGE_IN_LINUX')

IS_SHORT = True

class prior:
    def to_julia_prior(self):
        pass

    def get_type(self):
        pass

    def to_JSON(self):
        pass


class niw(prior):
    def __init__(self, kappa, mu, nu, psi):
        if nu < len(mu):
            raise Exception('nu should be atleast the Dim')
        self.kappa = kappa
        self.mu = mu
        self.nu = nu
        self.psi = psi

    def to_julia_prior(self):
        return DPMMSubClusters.niw_hyperparams(self.kappa, self.mu, self.nu, self.psi)

    def get_type(self):
        return 'Gaussian'

    def to_JSON(self):
        j = {'k': self.kappa,
             'm': self.mu.tolist(),
             'v': self.nu,
             'psi': self.psi.tolist()
             }

        return j


class multinomial(prior):
    def __init__(self, alpha, dim=1):
        if isinstance(alpha, np.ndarray):
            self.alpha = alpha
        else:
            self.alpha = np.ones(dim) * alpha

    def to_julia_prior(self):
        return DPMMSubClusters.multinomial_hyper(self.alpha)

    def get_type(self):
        return 'Multinomial'

    def to_JSON(self):
        j = {'alpha': self.alpha.tolist()
             }

        return j


class DPMMPython:
    """
     Wrapper for the DPMMSubCluster Julia package
     """

    @staticmethod
    def create_niw_prior(dim, mean_prior, mean_str, cov_prior, cov_str):
        """
        Creates a gaussian prior, if cov_prior is a scalar, then creates an isotropic prior scaled to that, if its a matrix
        uses it as covariance
        :param dim: data dimension
        :param mean_prior: if a scalar, will create a vector scaled to that, if its a vector then use it as the prior mean
        :param mean_str: prior mean psuedo count
        :param cov_prior: if a scalar, will create an isotropic covariance scaled to cov_prior, if a matrix will use it as
        the covariance.
        :param cov_str: prior covariance psuedo counts
        :return: DPMMSubClusters.niw_hyperparams prior
        """
        if isinstance(mean_prior, (int, float)):
            prior_mean = np.ones(dim) * mean_prior
        else:
            prior_mean = mean_prior

        if isinstance(cov_prior, (int, float)):
            prior_covariance = np.eye(dim) * cov_prior
        else:
            prior_covariance = cov_prior
        prior = niw(mean_str, prior_mean, dim + cov_str, prior_covariance)
        return prior

    @staticmethod
    def create_mnmm_prior(alpha, dim):
        prior = multinomial(alpha, dim)
        return prior

    @staticmethod
    def fit(data, alpha, prior=None,
            iterations=100, verbose=False,
            burnout=15, gt=None, outlier_weight=0, outlier_params=None, gpu=True, force_kernel=2):
        """
        Wrapper for DPMMSubClusters fit, reffer to "https://bgu-cs-vil.github.io/DPMMSubClusters.jl/stable/usage/" for specification
        Note that directly working with the returned clusters can be problematic software displaying the workspace (such as PyCharm debugger).
        :return: labels, clusters, sublabels
        """
        if gpu == True:
            np.save("modelData.npy", np.swapaxes(data, 0, 1))

            modelParams = {'alpha': alpha,
                           'iterations': iterations,
                           'use_verbose': verbose,
                           'burnout_period': burnout,
                           'force_kernel': force_kernel,
                           'outlier_mod': outlier_weight,
                           'outlier_hyper_params': outlier_params,
                           'hyper_params': prior.to_JSON()
                           }
            if gt is not None:
                modelParams['gt'] = gt.tolist()

            with open('modelParams.json', 'w') as f:
                json.dump(modelParams, f)

            if platform.system().startswith('Windows'):
                process = subprocess.Popen([FULL_PATH_TO_PACKAGE_IN_WINDOWS,
                                            "--prior_type=" + prior.get_type(), "--model_path=modelData.npy",
                                            "--params_path=modelParams.json", "--result_path=result.json"])
            elif platform.system().startswith("Linux"):
                process = subprocess.Popen(
                    [FULL_PATH_TO_PACKAGE_IN_LINUX,
                     "--prior_type=" + prior.get_type(), "--model_path=modelData.npy", "--params_path=modelParams.json",
                     "--result_path=result.json"])
            else:
                print(f'Not support {platform.system()} OS')

            out, err = process.communicate()
            errcode = process.returncode

            process.kill()
            process.terminate()

            with open('result.json') as f:
                results_json = json.load(f)

            if "error" in results_json:
                print(f'Error:{results_json["error"]}')
                return [], []

            os.remove("result.json")
            return results_json["labels"], None, [results_json["weights"], results_json["iter_count"]]

        else:
            if prior == None:
                results = DPMMSubClusters.fit(data, alpha, iters=iterations,
                                              verbose=verbose, burnout=burnout,
                                              gt=gt, outlier_weight=outlier_weight,
                                              outlier_params=outlier_params)
            else:
                results = DPMMSubClusters.fit(data, prior.to_julia_prior(), alpha, iters=iterations,
                                              verbose=verbose, burnout=burnout,
                                              gt=gt, outlier_weight=outlier_weight,
                                              outlier_params=outlier_params)
            return results[0], results[1], results[2:]

    @staticmethod
    def get_model_ll(points, labels, clusters):
        """
        Wrapper for DPMMSubClusters cluster statistics
        :param points: data
        :param labels: labels
        :param clusters: vector of clusters distributions
        :return: vector with each cluster avg ll
        """
        return DPMMSubClusters.cluster_statistics(points, labels, clusters)[0]

    @staticmethod
    def add_procs(procs_count):
        j = julia.Julia()
        j.eval('using Distributed')
        j.eval('addprocs(' + str(procs_count) + ')')
        j.eval('@everywhere using DPMMSubClusters')
        j.eval('@everywhere using LinearAlgebra')
        j.eval('@everywhere BLAS.set_num_threads(2)')

    @staticmethod
    def generate_gaussian_data(sample_count, dim, components, var):
        '''
        Wrapper for DPMMSubClusters cluster statistics
        :param sample_count: how much of samples
        :param dim: samples dimension
        :param components: number of components
        :param var: variance between componenets means
        :return: (data, gt)
        '''
        data = DPMMSubClusters.generate_gaussian_data(sample_count, dim, components, var)
        gt = data[1]
        data = data[0]
        return data, gt

    @staticmethod
    def generate_mnmm_data(sample_count, dim, components, trials):
        '''
        Wrapper for DPMMSubClusters cluster statistics
        :param sample_count: how much of samples
        :param dim: samples dimension
        :param components: number of components
        :param trials: draws from each vector
        :return: (data, gt)
        '''
        data = DPMMSubClusters.generate_mnmm_data(sample_count, dim, components, trials)
        gt = data[1]
        data = data[0]
        return data, gt

def generate_gaussian_data(n_samples, d, k):
    if d > 4:
        return DPMMPython.generate_gaussian_data(n_samples, d, k, 0.1)
    else:
        return DPMMPython.generate_gaussian_data(n_samples, d, k, 100)


def generate_mnmm_data(n_samples, d, k):
    print(f'start generate_mnmm_data: {datetime.now()}')
    return DPMMPython.generate_mnmm_data(n_samples, d, k, 50)


def generate_mnist_data(n_samples, d, k):
    data = np.load('mnist_images.npy')
    pca = PCA(n_components=d)
    data = pca.fit(data).transform(data)
    data = data - data.mean(axis=0)
    data = data / data.std(axis=0)
    data = np.swapaxes(data, 0, 1)
    gt = np.load('mnist_labels.npy').flatten()
    return data, gt


def generate_fashion_mnist_data(n_samples, d, k):
    data = np.load('fashion_mnist_images.npy')
    pca = PCA(n_components=d)
    data = pca.fit(data).transform(data)
    data = data - data.mean(axis=0)
    data = data / data.std(axis=0)
    data = np.swapaxes(data, 0, 1)
    gt = np.load('fashion_mnist_labels.npy')
    return data, gt


def generate_imagenet64_data(n_samples, d, k):
    data = np.load('imagenet64_images.npy')
    data = np.swapaxes(data, 0, 1)
    gt = np.load('imagenet64_labels.npy')
    return data, gt


def generate_20newsgroups20k_data(n_samples, d, k):
    data = np.load('20newsgroups20000_train.npy')
    data = np.swapaxes(data, 0, 1)
    gt = np.load('20newsgroups20000_labels.npy')
    return data, gt


def run_test(n_samples, d, k, numIter=10, max_iter=100, model='',
             get_data=generate_gaussian_data, prior=None, prior_niw_if_none=True, alpha=1, burnout=15,
             force_kernel=0, run_julia=True, run_cuda=True, run_sklearn=True):
    print(f'n_samples={n_samples}, d={d}, k={k}, numIter={numIter}, model={model}: {datetime.now()}')
    # Generate sample
    data, gt = get_data(n_samples, d, k)
    if prior == None:
        if prior_niw_if_none:
            prior = DPMMPython.create_niw_prior(d, 0, 1, 1, 1)
        else:
            prior = DPMMPython.create_mnmm_prior(1, d)

    df = pd.DataFrame()
    df.index.name = 'Iter'

    params_str = str(n_samples) + '_' + str(d) + '_' + str(k)
    for i in range(numIter):
        if run_julia:
            # Julia
            print(f'Julia...: {datetime.now()}')
            labels, _, more = DPMMPython.fit(data, alpha, iterations=max_iter, prior=prior, verbose=False,
                                             burnout=burnout, gt=gt, gpu=False)
            nmi_result = nmi(gt, labels)
            print(f'NMI:{nmi_result}')
            df['Time_elapse_' + params_str + '_Julia' + str(i)] = more[1]
            df['NMI_' + params_str + '_Julia' + str(i)] = nmi_result

        if run_cuda:
            # Cuda
            print(f'Cuda...: {datetime.now()}')
            labels, _, more = DPMMPython.fit(data, alpha, iterations=max_iter, prior=prior, verbose=True,
                                             burnout=burnout, gt=gt, gpu=True, force_kernel=force_kernel)
            nmi_result = nmi(gt, labels)
            print(f'NMI:{nmi_result}')
            df['NMI_' + params_str + '_Cuda' + str(i)] = nmi_result
            df['Time_elapse_' + params_str + '_Cuda' + str(i)] = more[1]

        if run_sklearn:
            # Sklearn GM
            print(f'Sklearn_GM......: {datetime.now()}')
            gm = GaussianMixture(n_components=k, random_state=0, max_iter=max_iter, verbose=0, verbose_interval=1000)
            tic = time()
            gm.fit(data.T)
            gmm_time = time() - tic
            labels_pred = gm.predict(data.T)
            nmi_result = nmi(gt, labels_pred)
            print(f'NMI:{nmi_result}')
            df['NMI_' + params_str + '_Sklearn_GM' + str(i)] = nmi_result
            df['Time_elapse_' + params_str + '_Sklearn_GM' + str(i)] = gmm_time

            # Sklearn BGM
            print(f'Sklearn_BGM......: {datetime.now()}')
            if i > 1 and d > 64:
                print('Skip on this iteration... too slow')
                continue
            gm = BayesianGaussianMixture(n_components=k * 5, random_state=0, max_iter=max_iter, verbose=0,
                                         verbose_interval=1000)
            tic = time()
            gm.fit(data.T)
            gmm_time = time() - tic
            labels_pred = gm.predict(data.T)
            nmi_result = nmi(gt, labels_pred)
            print(f'NMI:{nmi_result}')
            df['NMI_' + params_str + '_Sklearn_BGM' + str(i)] = nmi_result
            df['Time_elapse_' + params_str + '_Sklearn_BGM' + str(i)] = gmm_time

    path = os.path.join('results', 'run_result_' + model + '_' + params_str + '.csv')
    df = df.reindex(sorted(df.columns), axis=1)
    df.to_csv(path)

def calculate_nmi_mean(pd, str_to_search):
    filter_col = [col for col in pd if str_to_search in col and 'NMI' in col]
    sum_value = 0
    for i in range(len(filter_col)):
        sum_value += pd[filter_col[i]].iloc[-1]
    if len(filter_col) > 0:
        mean = sum_value / len(filter_col)
    else:
        mean = -1
    return mean


def calculate_time_mean(pd, str_to_search):
    filter_col = [col for col in pd if str_to_search in col and 'Time_elapse' in col]
    sum_value = 0
    for i in range(len(filter_col)):
        if 'Sklearn' in str_to_search:
            sum_value += pd[filter_col[i]].iloc[-1]
        else:
            sum_value += pd[filter_col[i]].sum(axis=0)
    if len(filter_col) > 0:
        mean = sum_value / len(filter_col)
    else:
        mean = -1
    return mean


def create_all_result_file(result_type, calculate_mean, model):
    columns_list = ['Params', 'Cuda', 'Julia', 'Sklearn_GM', 'Sklearn_BGM', 'X', 'D', 'K']
    df_all = pd.DataFrame(columns=columns_list)

    files = [f for f in listdir('results') if isfile(join('results', f))]

    for file in files:
        if 'run_result_' + model + '_' in file:
            pd_file = pd.read_csv(os.path.join('results', file), index_col=0)
            params = file.replace('.csv', '').replace('run_result_' + model + '_', '')
            params_list = params.split('_')
            new_row = pd.DataFrame([[params,
                                     calculate_mean(pd_file, 'Cuda'),
                                     calculate_mean(pd_file, 'Julia'),
                                     calculate_mean(pd_file, 'Sklearn_GM'),
                                     calculate_mean(pd_file, 'Sklearn_BGM'),
                                     params_list[0],
                                     params_list[1],
                                     params_list[2]
                                     ]], columns=columns_list)
            df_all = df_all.append(new_row, ignore_index=True)

    df_all = df_all.astype({'X': int, 'D': int, 'K': int})
    df_all = df_all.sort_values(by=['X', 'D', 'K'])
    path = os.path.join('results', 'run_result_all_' + model + '_' + result_type + '_table.csv')
    df_all.to_csv(path, index=False)

def buildDB():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape(60000, 784)
    np.save("mnist_images.npy",train_images)
    np.save("mnist_labels.npy",train_labels)

    if not IS_SHORT:
        url = "https://drive.google.com/uc?id=1_FgNQ5v9UnMSTbGduJjvue0a2p-EK2JZ"
        output = "imagenet_short.csv"
        gdown.download(url, output=output, quiet=False)

        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        train_images = train_images.reshape(60000, 784)
        np.save("fashion_mnist_images.npy",train_images)
        np.save("fashion_mnist_labels.npy",train_labels)

        cifar10 = tf.keras.datasets.cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        train_images = train_images.reshape(50000, 32*32*3)
        np.save("cifar10_images.npy",train_images)
        np.save("cifar10_labels.npy",train_labels)

        from numpy import genfromtxt
        data = genfromtxt('imagenet_short.csv', delimiter=',')
        np.save("imagenet64_images.npy",data[:,0:64])
        np.save("imagenet64_labels.npy",data[:,-1])
        print(data[:,-1])

        data, labels = fetch_20newsgroups_vectorized(subset='train',return_X_y=True,normalize=False)

        D = 20000
        print(data.shape)
        sum_row = data.sum(axis=0)
        sorted_sum_row = np.argsort(sum_row, axis=1)[0,::-1]
        indices = np.squeeze(np.asarray(sorted_sum_row[0,:D]))
        data_array = data[:,indices].toarray()
        np.save("20newsgroups"+str(D)+"_train.npy",data_array)
        np.save("20newsgroups"+str(D)+"_labels.npy",labels+1)


def run(is_short):
    global IS_SHORT
    IS_SHORT = is_short

    buildDB()
    if platform.system().startswith('Windows'):
        if FULL_PATH_TO_PACKAGE_IN_WINDOWS == None:
            print(
                'Missing path for windows package. For example: FULL_PATH_TO_PACKAGE_IN_WINDOWS = "C:/DPMMSubClusters.exe"')
            assert (False)
    elif platform.system().startswith("Linux"):
        if FULL_PATH_TO_PACKAGE_IN_LINUX == None:
            print(
                'Missing path for linux package. For example: FULL_PATH_TO_PACKAGE_IN_LINUX = "/home/user/bin/DPMMSubClusters"')
            assert (False)

    jl = Julia(compiled_modules=False)

    if not os.path.exists('results'):
        os.makedirs('results')

    # run known datasets
    D = 32
    K = 10
    if IS_SHORT:
        repeats = 1
    else:
        repeats = 10

    run_test(60000, D, K, repeats, max_iter=300, model='mnist', get_data=generate_mnist_data,
             prior=DPMMPython.create_niw_prior(D, 0, 1, 1.46, 456.8))
    if not IS_SHORT:
        run_test(60000, D, K, repeats, max_iter=200, model='fashion_mnist', get_data=generate_fashion_mnist_data,
                 prior=DPMMPython.create_niw_prior(D, 0, 1, 1.46, 456.8))
        run_test(125000, 64, 100, repeats, max_iter=200, model='imagenet64', get_data=generate_imagenet64_data,
                 prior=DPMMPython.create_niw_prior(64, 0, 1, 0.177459, 720.139))
        run_test(11314, 20000, 20, repeats, max_iter=100, model='20newsgroups10k', prior_niw_if_none=False,
                 get_data=generate_20newsgroups20k_data, force_kernel=2, run_sklearn=False)
    print(f'Complete test: {datetime.now()}')

    # generate gaussian random data

    max_iter = 100

    if IS_SHORT:
        repeats = 1
        N_range = [100000]
        D_range = [4]
        K_range = [4]
    else:
        repeats = 10
        N_range = [1000, 10000, 100000, 1000000]
        D_range = [2, 4, 8, 16, 32, 64, 128]
        K_range = [4, 8, 16, 32]

    for N in N_range:
        for D in D_range:
            for K in K_range:
                run_test(N, D, K, repeats, max_iter=max_iter, model='generated_gaussian',
                         get_data=generate_gaussian_data)
    # generate multinomial random data

    max_iter = 100
    if IS_SHORT:
        repeats = 1
        N_range = [100000]
        D_range = [4]
        K_range = [4]
    else:
        repeats = 10
        N_range = [1000, 10000, 100000, 1000000]
        D_range = [4, 8, 16, 32, 64, 128]
        K_range = [4, 8, 16, 32]

    for N in N_range:
        for D in D_range:
            for K in K_range:
                if D >= K:
                    run_test(N, D, K, repeats, max_iter=max_iter, model='generate_mnmm', prior_niw_if_none=False,
                             get_data=generate_mnmm_data, run_sklearn=False)

    create_all_result_file('NMI', calculate_nmi_mean, 'mnist')
    create_all_result_file('time', calculate_time_mean, 'mnist')
    create_all_result_file('NMI', calculate_nmi_mean, 'generated_gaussian')
    create_all_result_file('time', calculate_time_mean, 'generated_gaussian')
    create_all_result_file('NMI', calculate_nmi_mean, 'generate_mnmm')
    create_all_result_file('time', calculate_time_mean, 'generate_mnmm')
    if not IS_SHORT:
        create_all_result_file('NMI', calculate_nmi_mean, 'fashion_mnist')
        create_all_result_file('time', calculate_time_mean, 'fashion_mnist')
        create_all_result_file('NMI', calculate_nmi_mean, 'imagenet64')
        create_all_result_file('time', calculate_time_mean, 'imagenet64')
        create_all_result_file('NMI', calculate_nmi_mean, '20newsgroups10k')
        create_all_result_file('time', calculate_time_mean, '20newsgroups10k')
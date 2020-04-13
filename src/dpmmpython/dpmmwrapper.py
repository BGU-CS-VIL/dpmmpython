import julia
from priors import niw, multinomial
from julia import DPMMSubClusters
import numpy as np


class DPMMPython:
    """
     Wrapper for the DPMMSubCluster Julia package
     """

    @staticmethod
    def create_prior(dim,mean_prior,mean_str,cov_prior,cov_str):
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
        if isinstance(mean_prior,(int,float)):
            prior_mean = np.ones(dim) * mean_prior
        else:
            prior_mean = mean_prior

        if isinstance(cov_prior, (int, float)):
            prior_covariance = np.eye(dim) * cov_prior
        else:
            prior_covariance = cov_prior
        prior =niw(mean_str,prior_mean,dim + cov_str, prior_covariance)
        return prior

    @staticmethod
    def fit(data,alpha, prior = None,
            iterations= 100, verbose = False,
            burnout = 15, gt = None, outlier_weight = 0, outlier_params = None):
        """
        Wrapper for DPMMSubClusters fit, reffer to "https://bgu-cs-vil.github.io/DPMMSubClusters.jl/stable/usage/" for specification
        Note that directly working with the returned clusters can be problematic software displaying the workspace (such as PyCharm debugger).
        :return: labels, clusters, sublabels
        """
        if prior == None:
            results = DPMMSubClusters.fit(data,alpha, iters = iterations,
                                          verbose = verbose, burnout = burnout,
                                          gt = gt, outlier_weight = outlier_weight,
                                          outlier_params = outlier_params)
        else:
            results = DPMMSubClusters.fit(data, prior.to_julia_prior(), alpha, iters=iterations,
                                          verbose=verbose, burnout=burnout,
                                          gt=gt, outlier_weight=outlier_weight,
                                          outlier_params=outlier_params)
        return results[0],results[1],results[-1]

    @staticmethod
    def get_model_ll(points,labels,clusters):
        """
        Wrapper for DPMMSubClusters cluster statistics
        :param points: data
        :param labels: labels
        :param clusters: vector of clusters distributions
        :return: vector with each cluster avg ll
        """
        return DPMMSubClusters.cluster_statistics(points,labels,clusters)[0]

    @staticmethod
    def add_procs(procs_count):
        j = julia.Julia()
        j.eval('using Distributed')
        j.eval('addprocs(' + str(procs_count) + ')')
        j.eval('@everywhere using DPMMSubClusters')
        j.eval('@everywhere using LinearAlgebra')
        j.eval('@everywhere BLAS.set_num_threads(2)')

    
    @staticmethod
    def generate_gaussian_data(sample_count,dim,components,var):
        '''
        Wrapper for DPMMSubClusters cluster statistics
        :param sample_count: how much of samples
        :param dim: samples dimension
        :param components: number of components
        :param var: variance between componenets means
        :return: (data, gt)
        '''
        data = DPMMSubClusters.generate_gaussian_data(sample_count, dim, components, var)
        gt =  data[1]
        data = data[0]
        return data,gt



if __name__ == "__main__":
    j = julia.Julia()
    data = DPMMPython.generate_gaussian_data(10000, 2, 10, 100.0)
    gt =  data[1]
    data = data[0]
    prior = DPMMPython.create_prior(2, 0, 1, 1, 1)
    labels,_,sub_labels= DPMMPython.fit(data,100,prior = prior,verbose = True, gt = gt)
    prior = 0
    _ = 0
    print labels
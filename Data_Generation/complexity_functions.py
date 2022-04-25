from enum import Enum
import rpy2.robjects as robjects
from data_generator import generate_data
from functools import partial

r = robjects.r
r['source']('../Complexity/ECoL_complexity_functions.r')
n_points = 100

# GA constraint us to pass only single input X, so we cannot pass options for complexity
# for this reason we have to make specific function for each measure --> alot of repetition
# in final exp we no longer use GA
# TODO update the code for PSO, so we can reduce redundancy

def L1(X):
    r_from_pd_df = generate_data(n_points, X, r=True)
    function_r = robjects.globalenv['linearity_complexity_check']
    result_r = function_r(r_from_pd_df, 'L1')
    complex_value = result_r.rx()[0][0]
    return complex_value


def L2(X):
    r_from_pd_df = generate_data(n_points, X, r=True)
    function_r = robjects.globalenv['linearity_complexity_check']
    result_r = function_r(r_from_pd_df, 'L2')
    complex_value = result_r.rx()[0][0]
    return complex_value


def N1(X):
    r_from_pd_df = generate_data(n_points, X, r=True)
    function_r = robjects.globalenv['neighborhood_complexity_check']
    result_r = function_r(r_from_pd_df, 'N1')
    complex_value = result_r.rx()[0][0]
    return complex_value


def N2(X):
    r_from_pd_df = generate_data(n_points, X, r=True)
    function_r = robjects.globalenv['neighborhood_complexity_check']
    result_r = function_r(r_from_pd_df, 'N2')
    complex_value = result_r.rx()[0][0]
    return complex_value


def T2(X):
    r_from_pd_df = generate_data(n_points, X, r=True)
    function_r = robjects.globalenv['dimensionality_complexity_check']
    result_r = function_r(r_from_pd_df, 'T2')  # convert to float
    complex_value = result_r.rx()[0]
    return complex_value


def C1(X):
    r_from_pd_df = generate_data(n_points, X, r=True)
    function_r = robjects.globalenv['balance_complexity_check']
    result_r = function_r(r_from_pd_df, 'C1')  # convert to float
    complex_value = result_r.rx()[0][0]
    return complex_value


def F1(X):
    r_from_pd_df = generate_data(n_points, X, r=True)
    function_r = robjects.globalenv['overlapping_complexity_check']
    result_r = function_r(r_from_pd_df, 'F1')  # convert to float
    complex_value = result_r.rx()[0][0]
    return complex_value


def F2(X):
    r_from_pd_df = generate_data(n_points, X, r=True)
    function_r = robjects.globalenv['overlapping_complexity_check']
    result_r = function_r(r_from_pd_df, 'F2')  # convert to float
    complex_value = result_r.rx()[0][0]
    return complex_value


def Density(X):
    r_from_pd_df = generate_data(n_points, X, r=True)
    function_r = robjects.globalenv['network_complexity_check']
    result_r = function_r(r_from_pd_df, 'Density')  # convert to float
    complex_value = result_r.rx()[0][0]
    return complex_value


def ClsCoef(X):
    r_from_pd_df = generate_data(n_points, X, r=True)
    function_r = robjects.globalenv['network_complexity_check']
    result_r = function_r(r_from_pd_df, 'ClsCoef')  # convert to float
    complex_value = result_r.rx()[0][0]
    return complex_value


class ComplexityFunction(Enum):
    L1 = L1
    L2 = L2
    N1 = N1
    N2 = N2
    T2 = T2
    C1 = C1
    F1 = F1
    F2 = F2
    Density = Density
    ClsCoef = ClsCoef

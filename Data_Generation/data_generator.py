import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.datasets import make_classification


def generate_data(n_samples, X, r=False):
    clusters = int(round(X[0]))
    if clusters < 1:
        clusters = 1
    else:
        clusters = clusters
    weight_c2 = 1 - X[3]
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=2,
                               n_clusters_per_class=int(round(clusters)), class_sep=int(X[1]), flip_y=X[2],
                               weights=[X[3], weight_c2], random_state=15)
    result = np.c_[X, y]
    if r:
        pd_df = pd.DataFrame(result)
        pd_df_indexed = pd_df.iloc[list(range(1, len(pd_df.index))), :]
        with localconverter(robjects.default_converter + pandas2ri.converter):
            result = robjects.conversion.py2rpy(pd_df_indexed)
    return result


def validate_complexity(input_data, complex_spec = None):
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data)
    pd_df_indexed = input_data.iloc[list(range(1, len(input_data.index))), :]
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_from_pd_df = robjects.conversion.py2rpy(pd_df_indexed)
    if complex_spec is None:
        function_r = robjects.globalenv['all_complexities_check_subset']
        result_r = function_r(r_from_pd_df)
        complex_value = result_r.rx()
    else:
        function_r = robjects.globalenv[complex_spec[0]]
        result_r = function_r(r_from_pd_df, complex_spec[1])
        complex_value = result_r.rx()[0][0]
    return complex_value


def save_plot(dataset, file_name, value):
    sns.scatterplot(x=dataset[:, 0], y=dataset[:, 1], hue=dataset[:, 2], legend=False)
    plt.title('complexity_value : {}'.format(value))
    plt.savefig(file_name)
    plt.close()

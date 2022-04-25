from numpy import savetxt
from data_generator import validate_complexity
import numpy as np
import pandas as pd
import re
import os
import glob


dataset_list = ["C1", "N1", "L2", "F2"]

for current_dataset in dataset_list:

    path = '../Data_Generation/datasets/{}/batches'.format(current_dataset)
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    csv_files.sort()

    if current_dataset != 'C1': os.mkdir('../Data_Generation/datasets/{}/batches_C1_sort/'.format(current_dataset))
    if current_dataset != 'N1': os.mkdir('../Data_Generation/datasets/{}/batches_N1_sort/'.format(current_dataset))
    if current_dataset != 'L2': os.mkdir('../Data_Generation/datasets/{}/batches_L2_sort/'.format(current_dataset))
    if current_dataset != 'F2': os.mkdir('../Data_Generation/datasets/{}/batches_F2_sort/'.format(current_dataset))

    C1, N1, L2, F2 = ([] for i in range(4))

    for index, file in enumerate(csv_files):
        complexity = float(re.split('/', file)[-1].strip(".csv"))
        dataset = pd.read_csv(file)
        vals = validate_complexity(dataset)
        C1_val = complexity if current_dataset == 'C1' else vals[29]
        N1_val = complexity if current_dataset == 'N1' else vals[10]
        L2_val = complexity if current_dataset == 'L2' else vals[22]
        F2_val = complexity if current_dataset == 'F2' else vals[4]
        C1.append(C1_val)
        N1.append(N1_val)
        L2.append(L2_val)
        F2.append(F2_val)
        if current_dataset != 'C1':
            savetxt(
                '../Data_Generation/datasets/{}/batches_C1_sort/{}{}.csv'.format(current_dataset, np.round(C1_val, 5), index % 10),
                dataset, delimiter=',')
        if current_dataset != 'N1':
            savetxt(
                '../Data_Generation/datasets/{}/batches_N1_sort/{}{}.csv'.format(current_dataset, np.round(N1_val, 5), index % 10),
                dataset, delimiter=',')
        if current_dataset != 'L2':
            savetxt(
                '../Data_Generation/datasets/{}/batches_L2_sort/{}{}.csv'.format(current_dataset, np.round(L2_val, 5), index % 10),
                dataset, delimiter=',')
        if current_dataset != 'F2':
            savetxt(
                '../Data_Generation/datasets/{}/batches_F2_sort/{}{}.csv'.format(current_dataset, np.round(F2_val, 5), index % 10),
                dataset, delimiter=',')


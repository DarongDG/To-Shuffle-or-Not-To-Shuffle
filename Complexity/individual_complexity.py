import pandas as pd
import numpy as np
from sklearn import svm
import math

#import some flowers and remove semantics
def get_some_flowers():
    df = pd.read_csv('./data/iris.csv')
    df.rename(columns = {"Sepal.Length": "f1", "Sepal.Width": "f2", "Petal.Length": "f3", "Petal.Width": "f4", "Species": "class"}, inplace = True)
    for row in range(df.shape[0]):
        class_value = df["class"].iat[row]
        if class_value == "setosa":
            df["class"].iat[row] = 1
        elif class_value == "versicolor":
            df["class"].iat[row] = 2
        elif class_value == "virginica":
            df["class"].iat[row] = 3
    return df

#import numerical data
def get_generated_data(file_name):
    if ".xls" in file_name:
        df = pd.read_excel(file_name, header = None)
    else:
        df = pd.read_csv(file_name, header = None)
    for column_no in range(df.shape[1]-1):
        df.rename(columns = {list(df)[column_no]: "f"+str(column_no+1)}, inplace = True)
    df.rename(columns = {list(df)[df.shape[1]-1]: "class"}, inplace = True)
    df["class"] = df["class"].astype(int)+1
    pd.set_option('display.max_rows', None)
    return df

#separate into binary data sets
def make_binary_data_sets(df):
    num_classes = df["class"].max()
    binary_dfs = []
    for class_no in range(1, num_classes+1):
        current_binary_df = df.copy(deep = True)
        for row in range(df.shape[0]):
            class_value = current_binary_df["class"].iat[row]
            if class_value == class_no:
                current_binary_df["class"].iat[row] = 1
            else:
                current_binary_df["class"].iat[row] = 0
        binary_dfs.append(current_binary_df)
    return binary_dfs

#sort data by distance from the most separating hyperplanes for each on-versus-all data set
def order_by_linearity(df, keep_scores = False):
    binary_dfs = make_binary_data_sets(df)
    for binary_df_no in range(len(binary_dfs)):
        binary_df = binary_dfs[binary_df_no]
        x = np.array(binary_df.loc[:, binary_df.columns != "class"])
        y = np.array(binary_df.loc[:, binary_df.columns == "class"]).flatten().astype("int")
        hyperplane = svm.LinearSVC(dual = False) #find hyperplane with linear program
        hyperplane.fit(x, y)
        coefficients = hyperplane.coef_[0]
        binary_df.drop(binary_df.loc[binary_df['class'] == 0].index, inplace = True)
        binary_df["distance"] = 0.0
        denominator = 0
        for feature_no in range(coefficients.shape[0]): #find euclidean distance to hyperplane
            denominator += pow(coefficients[feature_no], 2)
        denominator = math.sqrt(denominator)
        for row in range(binary_df.shape[0]):
            numerator = 0
            for feature_no in range(coefficients.shape[0]):
                numerator += coefficients[feature_no]*binary_df.iat[row, feature_no]
            numerator += coefficients[coefficients.shape[0]-1]
            binary_df["distance"].iat[row] = numerator/denominator
        total_distance = binary_df["distance"].sum()
        if total_distance < 0: #set data points on the right side to positive and the wrong side to negative
            binary_df["distance"] = binary_df["distance"].mul(-1)
        binary_df["class"] = binary_df_no+1 #set correct class values and recombine binary data sets
    ordered_df = pd.concat(binary_dfs).sort_values(by = "distance", ascending = False)
    if not keep_scores:
        ordered_df = ordered_df.drop(columns = ["distance"])
    return ordered_df

#sort data by its presence in regions of inter-class overlap in terms of features
def order_by_feature(df, keep_scores = False):
    binary_dfs = make_binary_data_sets(df)
    regions = np.empty((len(binary_dfs)), dtype = object)
    for class_no in range(len(binary_dfs)): #define feature ranges for each class by minimum and maximum values
        binary_df = binary_dfs[class_no]
        binary_df.drop(binary_df.loc[binary_df['class'] == 0].index, inplace = True)
        for feature_no in range(binary_df.shape[1]-1):
            region = (binary_df.iloc[:, feature_no].min(), binary_df.iloc[:, feature_no].max())
            if regions[class_no] is None:
                regions[class_no] = []
            regions[class_no].append(region)
    densities = np.empty(binary_dfs[0].shape[1]-1, dtype = object)
    for feature_no in range(densities.shape[0]): #caculate class density in each region by counting overlapping ranges
        feature_bounds = []
        for class_no in range(len(binary_dfs)):
            feature_bounds.append(regions[class_no][feature_no])
        feature_densities = []
        lower, density = find_next_lowest_with_density(feature_bounds, float("-inf"))
        absolute_max = 0
        for class_no in range(regions.shape[0]):
            bounds = regions[class_no][feature_no]
            if bounds[1] > absolute_max:
                absolute_max = bounds[1]
        while lower < absolute_max:
            upper, density_change = find_next_lowest_with_density(feature_bounds, lower)
            feature_densities.append((lower, upper, density))
            density += density_change
            densities[feature_no] = feature_densities
            lower = upper
    for binary_df_no in range(len(binary_dfs)): #set correct class values and recombine data sets
        binary_dfs[binary_df_no]["class"] = binary_df_no+1
    df = pd.concat(binary_dfs)
    df["crowding"] = 0.0
    for row_no in range(len(df)): #score each data point by summing the densities of the regions in each feature dimension in which it falls
        crowding_score = 0
        for feature_no in range(df.shape[1]-2):
            feature_value = df.iat[row_no, feature_no]
            for region_density in densities[feature_no]:
                if region_density[0] < feature_value < region_density[1]:
                    crowding_score += region_density[2]
        df.iloc[row_no, df.columns.get_loc("crowding")] = crowding_score
    ordered_df = df.sort_values(by = "crowding")
    if not keep_scores:
        ordered_df = ordered_df.drop(columns = ["crowding"])
    return ordered_df

#sort data by ratio of intra-class distance sum to inter-class distance sum
def order_by_neighbourhood(df, keep_scores = False):
    df["ratio"] = 0.0
    for datum_row_no in range(len(df)): #sum inter-class and intra-class distances
        datum_class = df["class"].iat[datum_row_no]
        inter = 0
        intra = 0
        for comparison_row_no in range(len(df)):
            comparison_class = df["class"].iat[comparison_row_no]
            distance = 0
            for feature_no in range(df.shape[1]-3):
                distance += pow(df.iloc[datum_row_no, feature_no]-df.iloc[comparison_row_no, feature_no], 2)
            distance = math.sqrt(distance)
            if datum_class == comparison_class:
                intra += distance
            else:
                inter += distance
        df["ratio"].iat[datum_row_no] = inter/intra
    ordered_df = df.sort_values(by = "ratio")
    if not keep_scores:
        ordered_df = ordered_df.drop(columns = ["ratio"])
    return ordered_df

#to streamline:
#add something to check that if the list is 0 its possible that there are 0 neighbours
#decrease threshold so that there are less edges
#sort data by connectivity and neighbour connectivity after graph pruning based on class
def order_by_network(df, keep_scores = False):
    threshold = abs(0.15*df.iloc[:, 0:df.shape[0]-2].max().max()-df.iloc[:, 0:df.shape[0]-2].min().min()) #0.15 as recommended by literature
    edges = []
    for datum_row_no in range(len(df)): #create network without the need for pruning, as that process adds nothing but computation resources here
        for comparison_row_no in range(len(df)):
            if df["class"].iat[datum_row_no] == df["class"].iat[comparison_row_no] and datum_row_no != comparison_row_no: #disallow edges connecting vetices to themselves or to vertices belonging to a different class
                distance = 0
                for feature_no in range(df.shape[1]-2):
                    distance += pow(df.iloc[datum_row_no, feature_no]-df.iloc[comparison_row_no, feature_no], 2)
                distance = math.sqrt(distance)
                if distance < threshold:
                    edge_no = 0
                    edge_already_exists = False
                    while edge_no < len(edges) and not edge_already_exists: #if an edge already exists, do not store it as order does not matter
                        if datum_row_no in edges[edge_no] and comparison_row_no in edges[edge_no]:
                            edge_already_exists = True
                        edge_no += 1
                    if not edge_already_exists:
                        edges.append((datum_row_no, comparison_row_no))
    df["hubs"] = 0.0 #calculate Hubs score
    df["neighbours"] = np.empty((len(df), 0)).tolist() #streamline by preventing recounting
    df["checked"] = False
    count = 0
    for datum_row_no in range(len(df)):
        count += 1
        print("calculating", count)
        datum_connections_no = 0
        neighbour_connections_no = 0
        if len(df["neighbours"].iat[datum_row_no]) > 0:
            datum_connections_no = len(df["neighbours"].iat[datum_row_no])
            for neighbour in df["neighbours"].iat[datum_row_no]:
                if len(df["neighbours"].iat[neighbour]) > 0:
                    neighbour_connections_no += len(df["neighbours"].iat[neighbour])
                else:
                    for potential_neighbour_edge in edges:
                        if potential_neighbour_edge[0] == neighbour:
                            neighbour_connections_no += 1
                            df["neighbours"].iat[neighbour].append(potential_neighbour_edge[1])
                        elif potential_neighbour_edge[1] == neighbour:
                            neighbour_connections_no += 1
                            df["neighbours"].iat[neighbour].append(potential_neighbour_edge[0])
                    df["checked"].iat[neighbour] = True
        else:
            for edge in edges:
                if edge[0] == datum_row_no:
                    datum_connections_no += 1
                    df["neighbours"].iat[datum_row_no].append(edge[1])
                    if len(df["neighbours"].iat[edge[1]]) > 0:
                        neighbour_connections_no += len(df["neighbours"].iat[edge[1]])
                    else:
                        for potential_neighbour_edge in edges:
                            if edge[1] == potential_neighbour_edge[0]:
                                neighbour_connections_no += 1
                                df["neighbours"].iat[edge[1]].append(potential_neighbour_edge[1])
                            if edge[1] == potential_neighbour_edge[1]:
                                neighbour_connections_no += 1
                                df["neighbours"].iat[edge[1]].append(potential_neighbour_edge[0])
                        df["checked"].iat[edge[1]] = True
                elif edge[1] == datum_row_no:
                    datum_connections_no += 1
                    df["neighbours"].iat[datum_row_no].append(edge[0])
                    if len(df["neighbours"].iat[edge[0]]) > 0:
                        neighbour_connections_no += len(df["neighbours"].iat[edge[0]])
                    else:
                        for potential_neighbour_edge in edges:
                            if edge[0] == potential_neighbour_edge[0]:
                                neighbour_connections_no += 1
                                df["neighbours"].iat[edge[0]].append(potential_neighbour_edge[1])
                            elif edge[0] == potential_neighbour_edge[1]:
                                neighbour_connections_no += 1
                                df["neighbours"].iat[edge[0]].append(potential_neighbour_edge[0])
                        df["checked"].iat[edge[0]] = True
            df["checked"].iat[datum_row_no] = True
        hubs_score = neighbour_connections_no*datum_connections_no
        df["hubs"].iat[datum_row_no] = hubs_score
    ordered_df = df.sort_values(by = "hubs", ascending = False)
    ordered_df = ordered_df.drop(columns = ["neighbours", "checked"])
    if not keep_scores:
        ordered_df = ordered_df.drop(columns = ["hubs"])
    return ordered_df

#find the next point at which density changes and calculate by how much
def find_next_lowest_with_density(all_bounds, lower):
    upper = float("inf")
    density_change = 0
    for bounds in all_bounds:
        if bounds[0] == bounds[1]:
            if lower < bounds[0] < upper:
                density_change = 0
        else:
            if bounds[0] == upper:
                density_change += 1
            if bounds[1] == upper:
                density_change -= 1
            if lower < bounds[0] < upper:
                density_change = 1
                upper = bounds[0]
            elif lower < bounds[1] < upper:
                density_change = -1
                upper = bounds[1]
    return upper, density_change

#order provided data by specified method
def order_data(method, data_file, keep_scores = False, display = False):
    if data_file == "Iris":
        print("ordering Iris data")
        data = get_some_flowers()
    else:
        try:
            data = get_generated_data(data_file)
            print("ordering ", data_file, " data")
        except FileNotFoundError:
            print("data file not found, ordering Iris data")
            data = get_some_flowers()
    if method == "linearity":
        print("ordering by linearity")
        ordered = order_by_linearity(data, keep_scores = keep_scores)
    elif method == "feature":
        print("ordering by feature")
        ordered = order_by_feature(data, keep_scores = keep_scores)
    elif method == "network":
        print("ordering by network")
        ordered = order_by_network(data, keep_scores = keep_scores)
    else:
        print("ordering by neighbourhood")
        ordered = order_by_neighbourhood(data, keep_scores = keep_scores)
    if display:
        print(ordered)
    return ordered

order_data("neighbourhood", "Iris", True) #pass an empty string for Iris data

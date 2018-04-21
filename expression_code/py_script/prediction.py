import h5py
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn import datasets

def load_data():

    mat = h5py.File('expression_code/data/processed_ck.mat')

    # Transformation Matrix (neutral, expression, neutral, expression)
    def_coeff = np.array(mat["def_coeff"])

    # Loading expression labels
    labels_expr = []
    with h5py.File('expression_code/data/processed_ck.mat') as f:
        column = f['labels_expr'][0]
        for row_number in range(len(column)):
            labels_expr.append(''.join(map(chr, f[column[row_number]][:])))

    labels_expr = np.asarray(labels_expr)

    # Computing expressions and dataset dictionary
    expressions_dict = {}
    dataset_dict = {}

    for i in range(1, def_coeff.shape[0], 2):

        neutral = def_coeff[i - 1, :]
        expr = def_coeff[i, :]
        expr = np.matrix(expr)
        neutral = np.matrix(neutral)
        label = labels_expr[i]

        if label in expressions_dict:
            expressions_dict[label] = np.append(expressions_dict[label], expr, axis=0)
        else:
            expressions_dict[label] = expr

        #Computing dataset for machine learning for each expression
        if label in dataset_dict:
            dataset_dict[label]["input"] = np.append(dataset_dict[label]["input"], neutral, axis=0)
            dataset_dict[label]["output"] = np.append(dataset_dict[label]["output"], expr, axis=0)
        else:
            dataset_dict[label] = {"input" : neutral, "output" : expr}

    return expressions_dict, dataset_dict

# This method computes mean/median/mode of a particular expression to predict the related transformation
def m_prediction(expr = 'happy', tec ='mean', bandwidth = None):

    expressions_dict, dataset_dict = load_data()

    if tec == 'median':
        # Computing median
        return np.median(expressions_dict[expr], axis=0)

    elif tec == 'mode':
        # Computing mode
        if bandwidth != None:

            # If bandwidth is specified, return the centroid with max density

            delta = expressions_dict[expr]
            ms = MeanShift(bandwidth)
            ms.fit(delta)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_

            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            print("In radius ", bandwidth, " we get ", n_clusters_, " centroids")

            if (len(cluster_centers) > 1):

                for index, cluster_center in enumerate(cluster_centers):
                    print("For cluster with index ", index)

                    cluster_distance = euclidean_distances(cluster_center.reshape(1, -1), cluster_centers)
                    print("Distance between centroid and other centroids: ", cluster_distance[0])

                    num_vector = len(np.where(labels == index)[0])
                    print("Number of vectors within the radius centered in the first centroid: ",
                          num_vector, " ", num_vector / len(labels) * 100, "%")

            return cluster_centers[0]

        # If bandwidth is not specified, repeat until there is only a centroid, and return it
        n_clusters_ = 0
        quantile = 0.3
        while n_clusters_ != 1:
            delta = expressions_dict[expr]

            bandwidth = estimate_bandwidth(delta, quantile=quantile)
            ms = MeanShift(bandwidth=bandwidth)
            ms.fit(delta)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_

            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            quantile += 0.1

        return cluster_centers[0]
    else:
        # Computing mean
        return np.mean(expressions_dict[expr], axis=0)

# This method return a Linear/SVR/Neural Network Regressor for a particular expression
# to predict the related transformation
def regressor(expr, tec = "svr", kernel = "rbf", cv_test = False, cv_array = 10, learning_rate = 0.01):
    # Load and split data
    expressions_dict, dataset_dict = load_data()

    X = dataset_dict[expr]["input"]
    y = dataset_dict[expr]["output"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print("")
    print("Training the " + tec + " regressor for expression ", expr ,"...")
    if tec == "linear":
        # Create Linear Regressor
        regr = LinearRegression()
        regr.fit(X_train, y_train)

    elif tec == "svr":
        # Create SVR Regressos
        best_parameters = {"C": 200, "gamma": 0.1, "score": -10000, "n_split": 4}

        if(cv_test):
            # If cv_test = True, making handcrafted cross validation for parameters C and gamma (it can take a while)
            print("Starting cross validation ...")
            c_array = np.geomspace(100.0, 300.0, cv_array)
            gamma_array = np.geomspace(0.1, 1000.0, cv_array)

            # Validate the best number of splits
            for split in range(2,12,2):
                scores = []
                scores = np.array(scores)
                kfold = KFold(n_splits=split, random_state = None, shuffle=True)
                for train_index, test_index in kfold.split(X):
                    regr = MultiOutputRegressor(SVR(kernel=kernel, gamma=200, C=1))
                    kf_x_train, kf_x_test = X[train_index], X[test_index]
                    kf_y_train, kf_y_test = y[train_index], y[test_index]
                    regr.fit(kf_x_train, kf_y_train)
                    kf_y_pred = regr.predict(kf_x_test)
                    scores = np.append(scores, r2_score(kf_y_test, kf_y_pred))

                if best_parameters["score"] < np.mean(scores):
                    best_parameters["score"] = np.mean(scores)
                    best_parameters["n_split"] = split

                print("CV mean score for ", split ," splits: ", np.mean(scores))

            kfold = KFold(n_splits = best_parameters["n_split"], random_state=None, shuffle=True)

            # Validate the best c and gamma parameters
            for c in c_array:
                for gamma in gamma_array:
                    scores = []
                    scores = np.array(scores)

                    for train_index, test_index in kfold.split(X):
                        regr = MultiOutputRegressor(SVR(kernel=kernel, gamma=gamma, C=c))
                        kf_x_train, kf_x_test = X[train_index], X[test_index]
                        kf_y_train, kf_y_test = y[train_index], y[test_index]
                        regr.fit(kf_x_train, kf_y_train)
                        kf_y_pred = regr.predict(kf_x_test)
                        scores = np.append(scores, r2_score(kf_y_test, kf_y_pred))

                    if np.mean(scores) > best_parameters["score"]:
                        best_parameters["score"] = np.mean(scores)
                        best_parameters["C"] = c
                        best_parameters["gamma"] = gamma


            print("For expr ", expr, " best parameters are ", best_parameters)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/best_parameters["n_split"], random_state=42)

        # Create a MultiOutputRegressor SVR
        regr = MultiOutputRegressor(SVR(kernel=kernel, gamma=best_parameters["gamma"], C=best_parameters["C"]))
        regr.fit(X_train, y_train)

    elif tec == "nn":

        best_parameters = {"n_split": 4, "score": -1000}

        if(cv_test):

            # Validate the best number of splits
            for split in range(2,12,2):
                scores = []
                scores = np.array(scores)
                kfold = KFold(n_splits=split, random_state = None, shuffle=True)
                for train_index, test_index in kfold.split(X):
                    regr = create_network(learning_rate)
                    kf_x_train, kf_x_test = X[train_index], X[test_index]
                    kf_y_train, kf_y_test = y[train_index], y[test_index]
                    regr.fit(kf_x_train, kf_y_train, epochs=20, verbose=0)
                    kf_y_pred = regr.predict(kf_x_test)
                    scores = np.append(scores, r2_score(kf_y_test, kf_y_pred))

                if best_parameters["score"] < np.mean(scores):
                    best_parameters["score"] = np.mean(scores)
                    best_parameters["n_split"] = split

                print("CV mean score for ", split ," splits: ", np.mean(scores))
                print("CV variance score for ", split, " splits: ", np.var(scores))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 / best_parameters["n_split"],
                                                                    random_state = 42)

        regr = create_network(learning_rate)

        regr.fit(X_train, y_train, epochs = 20, verbose=0)

    print("Regressor trained!")
    # Compute and print RMSE
    y_pred = regr.predict(X_test)

    print("")
    errors_v = []
    errors_v = np.array(errors_v)
    for (index, y_pred_v) in enumerate(y_pred):
        errors_v = np.append(errors_v, np.sqrt(mean_squared_error(y_pred_v.reshape(1,-1), y_test[index])))

    print("Accuracy and error for ", tec, " regression: ")
    print("Mean mean squared errors: ", np.mean(errors_v))
    print("Variance mean quared errors: ", np.var(errors_v))

    rmse = np.sqrt(mean_squared_error(y_pred, y_test))
    print("Root Mean Squared Error: {}".format(rmse))
    print("")

    return regr

def create_network(learning_rate):
    # Create neural network model
    regr = Sequential()

    regr.add(Dense(units=300, activation='relu', input_dim=300))

    regr.add(Dense(units=512, activation='tanh'))  # hidden layer

    regr.add(Dense(units=300, activation='relu'))

    learning_rate = learning_rate
    optimizer = keras.optimizers.Adam(lr=learning_rate)

    regr.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return regr

if __name__ == '__main__':

    # Mean Shift Study

    mat = h5py.File('../data/processed_ck.mat')

    # Transformation Matrix (neutral, expression, neutral, expression)
    def_coeff = np.array(mat["def_coeff"])

    # Ids of faces
    labels_id = np.array(mat["labels_id"])

    # Number of different faces
    face_number = np.max(labels_id)

    # Loading expression labels
    labels_expr = []
    with h5py.File('../data/processed_ck.mat') as f:
        column = f['labels_expr'][0]
        for row_number in range(len(column)):
            labels_expr.append(''.join(map(chr, f[column[row_number]][:])))

    # Expression labels
    labels_expr = np.asarray(labels_expr)

    # Different types of expression
    different_expr = np.sort(np.unique(labels_expr))

    # Computing expressions matrix
    expressions_dict = {}
    for i in range(1, def_coeff.shape[0], 2):

        trans = def_coeff[i, :] - def_coeff[i - 1, :]
        trans = np.matrix(trans)
        label = labels_expr[i]
        if label in expressions_dict:
            expressions_dict[label] = np.append(expressions_dict[label], trans, axis=0)
        else:
            expressions_dict[label] = trans

    radius = []
    radius = np.array(radius)

    for expr in different_expr:
        expressions_dict[expr] = def_coeff[np.where(labels_expr == expr)]
        # Computing modes
        n_clusters_ = 1000
        quantile = 0.2
        if expr == 'neutral':
            continue
        print("For expression ", expr)

        delta = expressions_dict[expr]

        bandwidth = estimate_bandwidth(delta, quantile=quantile)
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(delta)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        print("In radius ", bandwidth, " we get ", n_clusters_, " centroids")

        if(len(cluster_centers) > 1):

            for index, cluster_center in enumerate(cluster_centers):
                print("For cluster with index ", index)

                cluster_distance = euclidean_distances(cluster_center.reshape(1, -1), cluster_centers)
                print("Distance between centroid and other centroids: ", cluster_distance[0])

                num_vector = len(np.where(labels == index)[0])
                print("Number of vectors within the radius centered in the first centroid: ",
                      num_vector, " ", num_vector / len(labels) * 100, "%")

        print("")
from expression_code.py_script import prediction
from scipy.stats import pearsonr
import numpy as np
import h5py
import pickle
import sys


def predict(expr):
    mat = h5py.File('expression_code/data/processed_ck.mat')

    def_coeff = np.array(mat["def_coeff"])
    labels_expr = []

    with h5py.File('expression_code/data/processed_ck.mat') as f:
        column = f['labels_expr'][0]
        for row_number in range(len(column)):
            labels_expr.append(''.join(map(chr, f[column[row_number]][:])))

    labels_expr = np.asarray(labels_expr)
    def_neutral = def_coeff[labels_expr == 'neutral']

    regr = prediction.regressor(expr, tec="linear", corr=True)

    return regr


def sorting_index(vec, indexes):
    for i in range(1, len(vec)):
        tmp_vec = vec[i]
        tmp_ind = indexes[i]
        j = i - 1
        while j >= 0 and vec[j] > tmp_vec:
            vec[j+1] = vec[j]
            indexes[j + 1] = indexes[j]
            j = j - 1
        vec[j + 1] = tmp_vec
        indexes[j + 1] = tmp_ind


expr = input('digit expression between \nneutral, disgust, surprise, angry, sadness, fear, contempt, happy: \n')
if expr != "neutral":
    vec_linear_regression = predict(expr)
    # print(vec_linear_regression)
    weight = np.zeros([300, 1])
    for i in range(0, 300):
        weight[i] = np.sum(vec_linear_regression[i][:])

    # threshold value
    value = 0.7

    vec_index = np.where(weight[:] >= value)
    vec_value = np.extract(weight[:] >= value, weight)

    sorting_index(vec_value, vec_index[0])
    # normalization
    print('Indexes are sorted in decreasing way of weight: \n', vec_index[0])
    maxx = np.amax(vec_value)
    vec_value = vec_value / maxx
    print('Values are normalized and sorted in decreasing way of weight: \n', vec_value)

else:

    index_id = []

    neutral_id = []
    happy_id = []

    count = 0

    # Read file matlab
    f = h5py.File('expression_code/data/processed_ck.mat', 'r')
    # Load all weights face
    def_coeff = f.get('def_coeff')
    def_coeff = np.transpose(np.array(def_coeff))
    labels_id = f.get('labels_id')
    labels_id = np.transpose(np.array(labels_id))
    expr = f.get('labels_expr')
    labels_expr = []
    for j in range(0, 654):
        st = expr[0][j]
        obj = f[st]
        labels_expr.append(''.join(chr(i) for i in obj[:]))

    # Build neutral vectors
    for i in range(0, len(labels_expr)):
        if labels_expr[i] == 'neutral':
            neutral_id.append(def_coeff[:, i])

    neutral_id = np.array(neutral_id)
    max_range_ne = []
    min_range_ne = []

    for i in range(0, neutral_id.shape[1]):
        max_range_ne.append(np.amax(neutral_id[:, i]))
        min_range_ne.append(np.amin(neutral_id[:, i]))

    max_range_ne = np.array(max_range_ne)
    min_range_ne = np.array(min_range_ne)

    distance_range_ne = abs(max_range_ne - min_range_ne)
    index_range_ne = []
    dis_ran_ne = []
    tmp_max = 1

    # threshold value
    thr = 50

    for i in range(0, distance_range_ne.shape[0]):
        if distance_range_ne[i] >= thr:
            dis_ran_ne.append(distance_range_ne[i])
            index_range_ne.append(i)
            if tmp_max < distance_range_ne[i]:
                tmp_max = distance_range_ne[i]
            # print('range: [', int(min_range[i]), int(max_range[i]), '] \nindex:', i)

    # normalize distance
    for j in range(0, len(dis_ran_ne)):
        dis_ran_ne[j] = dis_ran_ne[j] / tmp_max

    if dis_ran_ne:
        sorting_index(dis_ran_ne, index_range_ne)
        print('Indexes are sorted in decreasing way of weight: \n', index_range_ne)
        print('Values are normalized and sorted in decreasing way of weight: \n', dis_ran_ne)

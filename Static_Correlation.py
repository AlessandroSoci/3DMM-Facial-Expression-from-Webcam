''' Separete python file to calculate the Static Correlation (in this case we use Pearson correlation) between neutral
weights and expression weights '''


from scipy.stats import pearsonr
import numpy as np
import h5py

# Initialize variable

index_id = []

neutral_id = []
happy_id = []
disgust_id = []
surprise_id = []
angry_id = []
sadness_id = []
fear_id = []
contempt_id = []

neutral_happy_index = []
neutral_disgust_index = []
neutral_surprise_index = []
neutral_angry_index = []
neutral_sadness_index = []
neutral_fear_index = []
neutral_contempt_index = []

pearson_happy_array = np.array([])
pearson_disgust_array = np.array([])
pearson_surprise_array = np.array([])
pearson_sadness_array = np.array([])
pearson_angry_array = np.array([])
pearson_fear_array = np.array([])
pearson_contempt_array = np.array([])

pearson_list = []
count = 0
tmp = 0
ne = -1
ha = -1
di = -1
su = -1
an = -1
sa = -1
fe = -1
co = -1
tmp_ne = 1000
tmp_ha = 1000
tmp_di = 1000
tmp_su = 1000
tmp_an = 1000
tmp_sa = 1000
tmp_fe = 1000
tmp_co = 1000
ne_max = -1000
ha_max = -1000
su_max = -1000
sa_max = -1000
fe_max = -1000
co_max = -1000
di_max = -1000
an_max = -1000

# Read file matlab
f = h5py.File('data/processed_ck.mat', 'r')
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


# Modify database: one neutral for person
for i in range(1, 124):
    for j in range(0, len(labels_id)):
        if labels_id[j] == i:
            index_id.append(j)
    # print(i)
    # print(index_id)
    if index_id:
        z = min(index_id)+1
        while z <= max(index_id):
            if labels_expr[z-tmp] == 'neutral':
                labels_expr.pop(z-tmp)
                labels_id = np.delete(labels_id, z-tmp, 0)
                def_coeff = np.delete(def_coeff, z-tmp, 1)
                tmp = tmp + 1
                # print('FIND IT')
            z = z + 1
    # print('//////')
    count = count + 1
    tmp = 0
    # print(count)
    index_id = []


# print(len(labels_expr))
# print(labels_id.shape)
# print(def_coeff.shape)

# Build vectors for Pearson
for i in range(0, len(labels_expr)):
    if labels_expr[i] == 'neutral':
        neutral_id.append(def_coeff[:, i])
        tmp_ne = i
        ne = ne + 1
    elif labels_expr[i] == 'happy':
        happy_id.append(def_coeff[:, i])
        tmp_ha = i
        ha = ha + 1
    elif labels_expr[i] == 'disgust':
        disgust_id.append(def_coeff[:, i])
        tmp_di = i
        di = di + 1
    elif labels_expr[i] == 'contempt':
        contempt_id.append(def_coeff[:, i])
        tmp_co = i
        co = co + 1
    elif labels_expr[i] == 'angry':
        angry_id.append(def_coeff[:, i])
        tmp_an = i
        an = an + 1
    elif labels_expr[i] == 'fear':
        fear_id.append(def_coeff[:, i])
        tmp_fe = i
        fe = fe + 1
    elif labels_expr[i] == 'sadness':
        sadness_id.append(def_coeff[:, i])
        tmp_sa = i
        sa = sa + 1
    elif labels_expr[i] == 'surprise':
        surprise_id.append(def_coeff[:, i])
        tmp_su = i
        su = su + 1
    if tmp_ha != 1000:
        if labels_id[tmp_ne] == labels_id[tmp_ha]:
            neutral_happy_index.append([ne, ha])
            tmp_ha = 1000
    elif tmp_di != 1000:
        if labels_id[tmp_ne] == labels_id[tmp_di]:
            neutral_disgust_index.append([ne, di])
            tmp_di = 1000
    elif tmp_co != 1000:
        if labels_id[tmp_ne] == labels_id[tmp_co]:
            neutral_contempt_index.append([ne, co])
            tmp_co = 1000
    elif tmp_an != 1000:
        if labels_id[tmp_ne] == labels_id[tmp_an]:
            neutral_angry_index.append([ne, an])
            tmp_an = 1000
    elif tmp_fe != 1000:
        if labels_id[tmp_ne] == labels_id[tmp_fe]:
            neutral_fear_index.append([ne, fe])
            tmp_fe = 1000
    elif tmp_sa != 1000:
        if labels_id[tmp_ne] == labels_id[tmp_sa]:
            neutral_sadness_index.append([ne, sa])
            tmp_sa = 1000
    elif tmp_su != 1000:
        if labels_id[tmp_ne] == labels_id[tmp_su]:
            neutral_surprise_index.append([ne, su])
            tmp_su = 1000

l_ne = len(neutral_id)
l_ha = len(happy_id)
l_su = len(surprise_id)
l_sa = len(sadness_id)
l_fe = len(fear_id)
l_co = len(contempt_id)
l_di = len(disgust_id)
l_an = len(angry_id)

# find max
for t in range(0, l_ne):
    tmp_max = np.amax(neutral_id[t])
    if ne_max < tmp_max:
        ne_max = tmp_max
    if t < l_ha:
        tmp_max = np.amax(happy_id[t])
        if ha_max < tmp_max:
            ha_max = tmp_max
    if t < l_sa:
        tmp_max = np.amax(sadness_id[t])
        if sa_max < tmp_max:
            sa_max = tmp_max
    if t < l_su:
        tmp_max = np.amax(surprise_id[t])
        if su_max < tmp_max:
            su_max = tmp_max
    if t < l_fe:
        tmp_max = np.amax(fear_id[t])
        if fe_max < tmp_max:
            fe_max = tmp_max
    if t < l_co:
        tmp_max = np.amax(contempt_id[t])
        if co_max < tmp_max:
            co_max = tmp_max
    if t < l_an:
        tmp_max = np.amax(angry_id[t])
        if an_max < tmp_max:
            an_max = tmp_max
    if t < l_di:
        tmp_max = np.amax(disgust_id[t])
        if di_max < tmp_max:
            di_max = tmp_max

# normalize all
neutral_id[:] = [x / ne_max for x in neutral_id]
happy_id[:] = [x / ha_max for x in happy_id]
sadness_id[:] = [x / sa_max for x in sadness_id]
surprise_id[:] = [x / su_max for x in surprise_id]
fear_id[:] = [x / fe_max for x in fear_id]
contempt_id[:] = [x / co_max for x in contempt_id]
angry_id[:] = [x / an_max for x in angry_id]
disgust_id[:] = [x / di_max for x in disgust_id]

# convert all list in numpy array
neutral_id = np.array(neutral_id)
happy_id = np.array(happy_id)
sadness_id = np.array(sadness_id)
surprise_id = np.array(sadness_id)
fear_id = np.array(fear_id)
contempt_id = np.array(contempt_id)
angry_id = np.array(angry_id)
disgust_id = np.array(disgust_id)

neutral_happy_index = np.array(neutral_happy_index)
neutral_angry_index = np.array(neutral_angry_index)
neutral_fear_index = np.array(neutral_fear_index)
neutral_sadness_index = np.array(neutral_sadness_index)
neutral_surprise_index = np.array(neutral_surprise_index)
neutral_disgust_index = np.array(neutral_disgust_index)
neutral_contempt_index = np.array(neutral_contempt_index)

# Pearson correlation
for p in range(0, 300):
    pearson_happy_array = np.append(pearson_happy_array, pearsonr(neutral_id[neutral_happy_index[:, 0], p],
                                                                  happy_id[neutral_happy_index[:, 1], p])[0])
    pearson_angry_array = np.append(pearson_angry_array, pearsonr(neutral_id[neutral_angry_index[:, 0], p],
                                                                  angry_id[neutral_angry_index[:, 1], p])[0])
    pearson_contempt_array = np.append(pearson_contempt_array, pearsonr(neutral_id[neutral_contempt_index[:, 0], p],
                                                                        contempt_id[neutral_contempt_index[:, 1], p])[0])
    pearson_disgust_array = np.append(pearson_disgust_array, pearsonr(neutral_id[neutral_disgust_index[:, 0], p],
                                                                      disgust_id[neutral_disgust_index[:, 1], p])[0])
    pearson_fear_array = np.append(pearson_fear_array, pearsonr(neutral_id[neutral_fear_index[:, 0], p],
                                                                fear_id[neutral_fear_index[:, 1], p])[0])
    pearson_sadness_array = np.append(pearson_sadness_array, pearsonr(neutral_id[neutral_sadness_index[:, 0], p],
                                                                      sadness_id[neutral_sadness_index[:, 1], p])[0])
    pearson_surprise_array = np.append(pearson_surprise_array, pearsonr(neutral_id[neutral_surprise_index[:, 0], p],
                                                                        surprise_id[neutral_surprise_index[:, 1], p])[0])

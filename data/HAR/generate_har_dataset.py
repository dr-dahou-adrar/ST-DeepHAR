import numpy as np
np.random.seed(1000)

from scipy.io import loadmat

har_dataset = r""

''' Loading the training samples '''
data_dict = loadmat(har_dataset + "UCI_HAR_DATASET.mat")
X_train_mat = data_dict['X_train'][0]
y_train_mat = data_dict['Y_train'][0]
X_test_mat = data_dict['X_test'][0]
y_test_mat = data_dict['Y_test'][0]

y_train = y_train_mat.reshape(-1, 1)
y_test = y_test_mat.reshape(-1, 1)

var_list = []
for i in range(X_train_mat.shape[0]):
    var_count = X_train_mat[i].shape[-1]
    var_list.append(var_count)

var_list = np.array(var_list)
max_nb_timesteps = var_list.max()
min_nb_timesteps = var_list.min()
median_nb_timesteps = np.median(var_list)

print('The maximum nb timesteps train : ', max_nb_timesteps)
print('The minimum nb timesteps train : ', min_nb_timesteps)
print('The median_nb_timesteps nb timesteps train : ', median_nb_timesteps)

X_train = np.zeros((X_train_mat.shape[0], X_train_mat[0].shape[0], max_nb_timesteps))

# pad terminating with zeros to obtain numpy arrays
for i in range(X_train_mat.shape[0]):
    var_count = X_train_mat[i].shape[-1]
    #print(i, X_train_mat[i])
    X_train[i, :, :var_count] = X_train_mat[i]

# ''' Loading the testing samples '''

X_test = np.zeros((X_test_mat.shape[0], X_test_mat[0].shape[0], max_nb_timesteps))

# '''pad terminating with zeros to obtain the numpy arrays'''
for i in range(X_test_mat.shape[0]):
    var_count = X_test_mat[i].shape[-1]
    X_test[i, :, :var_count] = X_test_mat[i][:, :max_nb_timesteps]


# ''' Save the datasets '''
print("Saving the training dataset : ", X_train.shape, y_train.shape)
print("Saving the testing dataset : ", X_test.shape, y_test.shape)
print("Saving the training dataset metrics : ", X_train.mean(), X_train.std())
print("Saving the testing dataset : ", X_test.mean(), X_test.std())
print("Nb classes : ", len(np.unique(y_train)))

np.save(har_dataset + 'X_train.npy', X_train)
np.save(har_dataset + 'y_train.npy', y_train)
np.save(har_dataset + 'X_test.npy', X_test)
np.save(har_dataset + 'y_test.npy', y_test)

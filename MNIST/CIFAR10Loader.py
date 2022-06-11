def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

base_path = "/home/whq//Downloads/"
test_dict = unpickle(base_path + 'cifar-10-batches-py/test_batch')

from heapq import merge
import numpy as np
import pandas as pd

data = test_dict[b'data']
label = test_dict[b'labels']

data = np.array(data)
label = np.array(label).reshape((len(label), 1))

merge_data = np.concatenate((data, label), axis=1)
print(merge_data.shape)

df = pd.DataFrame(merge_data)
df.to_csv(base_path + 'cifar10_test.csv', index=False, header=True)
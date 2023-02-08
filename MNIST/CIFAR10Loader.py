import numpy as np
import pandas as pd
import os


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


base_path = f"{os.path.expanduser('~')}/Downloads/"
if not os.path.exists(base_path):
    os.makedirs(base_path)

if os.path.exists(base_path + "cifar10_train.csv"):
    print(f'File already exists: {base_path + "cifar10_train.csv"}')
    exit()


# construct test set
test_dict = unpickle(base_path + "cifar-10-batches-py/test_batch")


data = test_dict[b"data"]
label = test_dict[b"labels"]

data = np.array(data)
label = np.array(label).reshape((len(label), 1))

test_data = np.concatenate((data, label), axis=1)
print(f"Train shape: {test_data.shape}")
print(f"Test: {test_data[0]}")


test_df = pd.DataFrame(test_data)
test_df.to_csv(base_path + "cifar10_test.csv", index=False, header=True)

# construct training set
batch_list = []
for i in range(1, 6):
    test_dict = unpickle(f"{base_path}cifar-10-batches-py/data_batch_{i}")

    data = test_dict[b"data"]
    label = test_dict[b"labels"]

    data = np.array(data)
    label = np.array(label).reshape((len(label), 1))

    merge_data = np.concatenate((data, label), axis=1)

    batch_list.append(merge_data)

train_data = np.concatenate(batch_list)
print(f"Train shape: {train_data.shape}")
print(f"Train: {train_data[0]}")

train_df = pd.DataFrame(train_data)
train_df.to_csv(base_path + "cifar10_train.csv", index=False, header=True)

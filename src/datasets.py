import numpy as np
import glob
from natsort import natsorted
from torch.utils.data import dataset, DataLoader
from sklearn.model_selection import train_test_split


def compile_data(file):
    probe_filez = natsorted(glob.glob(file))[0:3]

    probe_lfp = []
    for probefile in probe_filez:
        tmplfp = np.load(probefile)
        probe_lfp = np.append(probe_lfp, tmplfp[0])
    return probe_lfp


class LFP_Dataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label


class Dataset_Manager:
    def __init__(self, data, labels, train_size=0.8, shuffle=True):
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=1 - train_size, random_state=42, shuffle=shuffle
        )

        self.train_data = LFP_Dataset(X_train, y_train)
        self.test_data = LFP_Dataset(X_test, y_test)

    def get_train_loader(self, **args):
        return DataLoader(self.train_data, **args)

    def get_test_loader(self, **args):
        return DataLoader(self.test_data, **args)

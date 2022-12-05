import numpy as np
import torch
from torch import nn
from collections import namedtuple
from itertools import product
import sklearn.model_selection as sklearn

from DAL import DAL


class HWCRUtils:
    """
    This is a class with all utility methods.
    """
    @staticmethod
    def numpy_load(path, allow_pickle=False):
        """
        Loads data set.

        :param path: data set path
        :param allow_pickle:

        :return: the data set as numpy array
        """
        return np.load(path, allow_pickle=allow_pickle)

    @staticmethod
    def get_num_correct(preds, labels):
        """
        Calculates the number of the correct prediction.

        :param preds: predicted labels
        :param labels: true labels

        :return: total number of correctly predicted labels
        """
        return preds.argmax(dim=1).eq(labels).sum().item()

    @staticmethod
    def spilt_data_set(data_set, label_set, split_size):
        """
        Splits the data set into test and train set.

        :param data_set: dataset
        :param label_set: true labels
        :param split_size: split percentage

        :return: train and test dataset and corresponding labels
        """
        X_train, X_test, Y_train, Y_test = \
            sklearn.train_test_split(data_set, label_set, test_size=split_size, stratify=label_set)

        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def convert_to_tensor(X, Y, device):
        """
        Converts the dataset to tensor.

        :param X: dataset
        :param Y: label
        :param device: whether {cpu or gpu}

        :return: the dataset as tensor
        """
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_y = torch.from_numpy(Y)
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        return processed_dataset

    @staticmethod
    def read_dataset(data_set_path, label_set_path, image_dims):
        """
        Reads the dataset.

        :param data_set_path:
        :param label_set_path:
        :param image_dims:

        :return: dataset
        """
        dal = DAL()
        dal.read_data(data_set_path, label_set_path)
        train_data_set, labels_set = dal.pre_process_data_set(image_dims)
        return train_data_set, labels_set

    @staticmethod
    def read_dataset_test(data_set_path):
        """
        Reads the test set.
        :param data_set_path:

        :return: test set
        """
        dal = DAL()
        dal.read_data_test(data_set_path)
        data_set = dal.pre_process_data_set_test(64)
        return data_set

    @staticmethod
    def get_runs(params):
        """
        Gets the run parameters using cartesian products of the different parameters.

        :param params: different parameters like batch size, learning rates

        :return: iterable run set
        """
        Run = namedtuple("Run", params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

    @staticmethod
    def resize_padded(image, new_shape):
        """
        Resize each image of the data set to the dimension of 64 X 64 using padding.

        :param image: image in the data set
        :param new_shape: 64 X 64

        :return:  64 X 64 image
        """
        img = torch.from_numpy(image)
        delta_width = new_shape - img.shape[1]
        delta_height = new_shape - img.shape[0]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        if delta_width % 2 != 0:
            if delta_height % 2 != 0:
                pad = nn.ConstantPad2d((pad_width, pad_width + 1, pad_height, pad_height + 1), False)
            else:
                pad = nn.ConstantPad2d((pad_width, pad_width + 1, pad_height, pad_height), False)
        else:
            if delta_height % 2 != 0:
                pad = nn.ConstantPad2d((pad_width, pad_width, pad_height, pad_height + 1), False)
            else:
                pad = nn.ConstantPad2d((pad_width, pad_width, pad_height, pad_height), False)
        return pad(img).numpy()

    @staticmethod
    def get_device():
        """
        Gets the hardware device to use (cuda, mps or cpu)

        :return: device
        """
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_data(data_dict, plots_dir_path):
    for x in data_dict:
        title = x['title']
        data = x['data']
        plt.plot(data)
        plt.grid(True)
        plt.title(title)
        plt.ylabel(x['y_label'])
        plt.xlabel(x['x_label'])
        plt.savefig(os.path.join(plots_dir_path, title + ".png"))
        plt.clf()

    print('Plot Saved!')


def ensure_directory_exits(directory_path):
    """creates directory if path doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    # out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    out *= std / torch.sqrt(out.pow(2).unsqueeze(0).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0.1)

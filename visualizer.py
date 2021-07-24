import numpy as np
import matplotlib.pyplot as plt
import torch
import utils
from collections import defaultdict


def plot_bottleneck_representation(bottleneck_representation, y):
    bottleneck_x, bottleneck_y = torch.transpose(bottleneck_representation)
    plt.scatter(bottleneck_x, bottleneck_y, c=y)
    plt.show()


def show_b2_model_hidden_representation(model, x, y, bottleneck_layer_name='bottleneck'):
    y = utils.one_hot_to_index_vector(y)
    bottleneck_representation = utils.get_activation_from_layer(
        model, bottleneck_layer_name, x)

    bottleneck_rep_0_4, y_0_4 = utils.get_hidden_representation_for_class(
        bottleneck_representation, y, range(5), 250)
    bottleneck_rep_0_9, y_0_9 = utils.get_hidden_representation_for_class(
        bottleneck_representation, y, range(10), 250)

    plot_bottleneck_representation(bottleneck_rep_0_4, y_0_4)
    plot_bottleneck_representation(bottleneck_rep_0_9, y_0_9)


def compare_svd_for_b12_models(models, x, y, classes=range(10), bottleneck_layer_name='bottleneck'):
    y = utils.one_hot_to_index_vector(y)
    class_to_svd = defaultdict(dict)
    for model in models:
        bottleneck_representation = utils.get_activation_from_layer(
            model, bottleneck_layer_name, x)
        for c in classes:
            bottleneck_rep_c, _ = utils.get_hidden_representation_for_class(
                bottleneck_representation, y, [c])
            s, _, _ = torch.linalg.svd(bottleneck_rep_c)
            class_to_svd[c][model.name] = s

    for c in classes:
        for model in models:
            plt.plot(class_to_svd[c][model.name], label=model.name)
        plt.legend()
        plt.title(
            "Singular Values of hidden representations for class {}".format(c))
        plt.show()


def plot_spiral_dataset(x, y, title=None, legend=True):
    if title:
        plt.title(title)
    one = x[y == 0, :]
    two = x[y == 1, :]
    plt.scatter(*zip(*one), c='deepskyblue', label='class 1')
    plt.scatter(*zip(*two), c='goldenrod', label='class 2')

    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    
    if legend:
        plt.legend()
    plt.show()


def plot_spiral_model_confidence(model, x_train, y_train, title='spiral model'):
    xi = np.arange(-15, 15, 0.1)
    xj = np.arange(-15, 15, 0.1)
    x_sample = np.array([[j, i] for i in xi for j in xj])
    out = model(torch.tensor(x_sample))
    out.shape

    # get P(Y=1|X)
    confidence = torch.transpose(torch.nn.functional.softmax(out, dim=1), 0, 1)[
        1].detach().numpy()
    confidence = confidence.reshape((len(xi), len(xj)))
    x, y = np.meshgrid(xi, xj)

    im = plt.pcolormesh(x, y, confidence)  # vmin=0, vmax=1
    plt.colorbar(im)

    plot_spiral_dataset(x_train, y_train, title, False)


def one_hot_to_index_vector(v):
    return np.argmax(v, axis=0)


def get_activation_from_layer(model, layer_name, inputs):
    activation = inputs
    for layer in model.layers:
        activation = layer(activation)
        if layer.name == layer_name:
            break
    return activation.numpy()


def get_hidden_representation_for_class(bottleneck_representation, y, c, subset=None):
    indices = [i for i, y_i in enumerate(y) if y_i in c]
    indices = indices[:subset] if subset else indices
    y = torch.gather(y, indices)
    bottleneck_representation = torch.gather(
        bottleneck_representation, indices)
    return bottleneck_representation, y

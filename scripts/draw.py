import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

models = ['SecureML', 'MiniONN', 'LeNet', 'AlexNet', 'VGG16']
metrics = ['test_acc', 'train_acc', 'loss']

font_family1 = {'family': 'Times New Roman', 'size': 24}
font_family2 = {'family': 'Times New Roman', 'size': 14}

# model_name = "SecureML"
interval = 20

def draw_acc_fig(x_full, y_full, x_quant, y_quant, x_plaintext, y_plaintext, name, base_path, model, dataset):
    length = len(x_quant)
    # length = 4000
    print("drawing")
    x_quant = x_quant[slice(0, length, interval)]
    y_quant = y_quant[slice(0, length, interval)]
    # x_full_BN = x_full_BN[slice(0, length, interval)]
    # y_full_BN = y_full_BN[slice(0, length, interval)]
    x_full = x_full[slice(0, length, interval)]
    y_full = y_full[slice(0, length, interval)]
    # x_plaintext_BN = x_plaintext_BN[slice(0, length, interval)]
    # y_plaintext_BN = y_plaintext_BN[slice(0, length, interval)]
    x_plaintext = x_plaintext[slice(0, length, interval)]
    y_plaintext = y_plaintext[slice(0, length, interval)]
    plt.cla()
    # fig = plt.figure()
    ax = plt.subplot()
    plt.plot(x_quant, y_quant, color='g', linestyle='-', lw=2)
    # plt.plot(x_full_BN, y_full_BN, color='purple', linestyle='-.', lw=2)
    plt.plot(x_full, y_full, color='purple', linestyle='--', lw=2)
    # plt.plot(x_plaintext_BN, y_plaintext_BN, color='black', linestyle='-.', lw=2)
    plt.plot(x_plaintext, y_plaintext, color='black', linestyle='-.', lw=2)
    plt.grid(True, alpha=0.5, linestyle='-.')
    # plt.yscale('log')
    plt.xlabel('Number of iterations', fontdict=font_family1)

    if name == 'acc':
        ylabel = 'Accuracy'
    elif name == 'loss':
        ylabel = 'Cross Entropy Loss'

    plt.ylabel(ylabel, fontdict=font_family1)
    plt.tick_params(labelsize=20)
    # plt.title(name, fontdict=font_family)
    # plt.xticks(x)
    plt.legend(['\textt{Ditto}', '2', 'Plain-text'], framealpha=1, prop=font_family2)
    plt.tight_layout()
    # plt.show()
    print(base_path + model + '_' + dataset + '_' + name + '.pdf')
    plt.savefig(base_path + model + '_' + dataset + '_' + name + '.pdf')

# def load_data(quant):
#     x = []
#     y = []
#     sufix = 'quant' if quant else 'full-precision'
#     base_path = 'log/' + model + '/' + sufix + "/"
#     file_path = base_path + metric + '.txt'
#     if os.path.exists(file_path):
#         file = open(file_path, 'r')
#         for line in file:
#             x.append(int(line.split('\t')[0]))
#             y.append(float(line.split('\t')[1]))
#     return x, y

def load_secret_data(is_mixed, model, dataset, suffix, use_BN=False, is_gpu=True):
    x = []
    y = []
    file_path = '../output/' + model + ' preloaded train_' + ('GPU' if is_gpu else 'CPU') + '_' + ('Mixed' if is_mixed else 'Full') + '_' + dataset + "_" + suffix + '.txt'
    print(file_path)
    # file_path = base_path + metric + '.txt'
    if os.path.exists(file_path):
        file = open(file_path, 'r')
        for line in file:
            x.append(int(line.split('\t')[0]))
            y.append(float(line.split('\t')[1]))
    print(len(x))
    return x, y

def load_plaintext_data(model, dataset, suffix, use_BN=True, is_gpu=True):
    x = []
    y = []
    file_path = '../output/' + model +  '_train_plaintext_' + dataset + '_' + suffix + '.txt'
    print(file_path)
    # file_path = base_path + metric + '.txt'
    if os.path.exists(file_path):
        file = open(file_path, 'r')
        for line in file:
            x.append(int(line.split('\t')[0]))
            y.append(float(line.split('\t')[1]))
    return x, y

if __name__ == '__main__':
    model = 'SecureML'
    dataset = 'MNIST'
    # suffix = 'acc'
    for suffix in ['acc', 'loss']:
        # x_full_BN, y_full_BN = load_secret_data(False, model, dataset, suffix, use_BN=True)
        x_full, y_full = load_secret_data(False, model, dataset, suffix, use_BN=False)
        x_mixed, y_mixed = load_secret_data(True, model, dataset, suffix)
        # x_plaintext_BN, y_plaintext_BN = load_plaintext_data(model, dataset, suffix, use_BN=True)
        x_plaintext, y_plaintext = load_plaintext_data(model, dataset, suffix, use_BN=False)
        print("sdasdasd")
        draw_acc_fig(x_full, y_full, x_mixed, y_mixed, x_plaintext, y_plaintext, suffix, '../output/', model, dataset)
    # for model in models:
    #         for metric in metrics:
    #             x_quant, y_quant = load_data(True)
    #             x_full, y_full = load_data(False)
    #             draw_acc_fig(x_quant, y_quant, x_full, y_full, metric, 'log/' + model + '/', model)
                # draw_acc_fig(, metric, 'log/' + model + '/')

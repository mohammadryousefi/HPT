import math
import glob
import pathlib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

def get_files(path):
    p = pathlib.Path(path)
    return glob.glob(str(p / "**/all_training_logs_in_one_file.csv"), recursive=True)

def get_keys(path):
    p = pathlib.Path(path)
    experiment = p.parts[-4].split('_')[-1]
    data_set = p.parts[-2]
    return experiment, data_set

def process(dir_path):
    files = get_files(dir_path)
    data_frames = {}
    for f in files:
        keys = get_keys(f)
        df = pd.read_csv(f, header=0)
        if keys[0] not in data_frames:
            data_frames[keys[0]] = {}
        if keys[1] not in data_frames[keys[0]]:
            data_frames[keys[0]][keys[1]] = {}
        for t in df.groupby('metric'):
            metric_key = t[0].split('_')[-1]
            data_frames[keys[0]][keys[1]][metric_key] = t[1]
            data_frames[keys[0]][keys[1]][metric_key].reset_index(drop=True, inplace=True)
    return data_frames

def make_plots(data_frames, show=True, save=False):
    for exp in data_frames:
        for data_set in data_frames[exp]:
            epochs = data_frames[exp][data_set]['loss']['step'].to_numpy()
            loss = data_frames[exp][data_set]['loss']['value'].to_numpy()
            for metric in data_frames[exp][data_set]:
                if metric == 'loss':
                    continue
                value = data_frames[exp][data_set][metric]['value'].to_numpy()
                plt.figure()
                cc = np.corrcoef(loss, value)[0, 1]
                title = f'{exp}-{data_set}\nCC: {cc:0.7f}'
                plt.title(title)
                plt.ylabel('Value')
                plt.xlabel('Epoch')
                plt.legend
                plt.plot(epochs, loss, epochs, value)
                plt.legend(['Obj. Func.', metric])
                if save:
                    plt.savefig(f'{exp}_{data_set}_loss_{metric}.png')
                if show:
                    plt.show()
                plt.close()


def make_obj_plot(data_frames, show=True, save=False, ylim=None, xlim=None, yscale=None, xscale=None):
    columns = ['step', 'value']
    data = []
    legends = []
    for exp in data_frames:
        for data_set in data_frames[exp]:
            data.append(data_frames[exp][data_set]['loss'][columns[0]].to_numpy())
            data.append(data_frames[exp][data_set]['loss'][columns[1]].to_numpy())
            legends.append(f'{exp}-{data_set}')
    try:
        plt.close()
    except:
        pass
    plt.figure()
    plt.plot(*data)
    plt.title('Obj. Func.')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(legends)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    if yscale is not None:
        plt.gca().set_yscale(yscale)
    if xscale is not None:
        plt.gca().set_xscale(xscale)
    if save:
        plt.savefig('ObjectiveValue.png')
    if show:
        plt.show()
    plt.close()

def smooth(arr, factor): # Implementation reproduced from tensorboard.
    last = 0
    arr = arr.flatten()
    sarr = np.zeros_like(arr)
    sarr[0] = arr[0]

    for i in range(len(arr)):
        last = last * factor + (1 - factor) * arr[i]
        db = 1 - math.pow(factor, i + 1)
        sarr[i] = last / db
    return sarr

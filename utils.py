import os

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy.io import loadmat

import setting_path as PATH


def recall_score(target, pred, average=None):
    if type(target) not in [list, tuple]:
        target = target.tolist()
        pred = pred.tolist()
    target_type = list(set(target))
    count_dict = {t_t: [0, 0] for t_t in target_type}
    for tar, pre in zip(target, pred):
        if tar == pre:
            count_dict[tar][0] += 1
        else:
            count_dict[tar][1] += 1
    count_dict = [c_d[0] / sum(c_d) for c_d in count_dict.values()]
    if average is not None:
        count_dict = sum(count_dict) / len(count_dict)
    return np.array(count_dict)


def pv_score(target, pred, average=None):
    if type(target) not in [list, tuple]:
        target = target.tolist()
        pred = pred.tolist()
    pred_type = list(set(pred))
    count_dict = {t_t: [0, 0] for t_t in pred_type}
    for tar, pre in zip(target, pred):
        if tar == pre:
            count_dict[pre][0] += 1
        else:
            count_dict[pre][1] += 1
    count_dict = [c_d[0] / sum(c_d) for c_d in count_dict.values()]
    if average is not None:
        count_dict = sum(count_dict) / len(count_dict)
    return np.array(count_dict)


def get_graph(base_record):
    row_data = load_data(base_record)
    row_data = row_data[0][0:360]
    row_time = [i / 360 for i in range(360)]

    lc = get_ls_signal(base_record)
    lc_data = lc[0][0:98, 0]
    lc_time = lc[1][0:98, 0]

    print(lc_data.shape)
    print(lc_time.shape)

    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111, xlabel="time(s)", ylabel="ECG(mv)")
    plt.plot(row_time, row_data, label="normal ADC")
    plt.plot(lc_time, lc_data, marker="o", markersize=3, label="level-cross ADC")
    ax.legend()
    plt.show()


def load_data(base_record, channel=0):  # [0, 1]
    record_name = os.path.join(PATH.mit_path, str(base_record))

    signals, _ = wfdb.rdsamp(record_name)

    annotation = wfdb.rdann(record_name, "atr")
    symbols = annotation.symbol
    positions = annotation.sample

    return signals[:, channel], symbols, positions


def get_ls_signal(base_record, channel=0):
    file_name = "Rec" + base_record + "_ED_ch" + str(channel + 1)
    path = os.path.join(PATH.mit_lc_path, file_name)

    data = loadmat(path)["edECG"]

    sig = data[0][0]
    dtype = data.dtype.fields
    ann_time = []
    ann_type = []
    for s, d in zip(sig, dtype.keys()):
        if d == ["ann"]:
            ann_time.append(s)
        elif d == ["anntype"]:
            ann_type.append(s)
        elif d == "RR":
            rr = s
        elif d == "counter":
            counter = s

    ann = [(ti, ty) for ti, ty in zip(ann_time, ann_type)]
    return sig[0], sig[1], rr, counter, ann


def minmax(signal):
    return max(signal), min(signal)


def main():
    val_dict = []

    record_list = ["101"]  # PATH.record_list

    for r in record_list:

        sig, time, rr, counter, ann = get_ls_signal(r)

        for s in sig[:, 0]:
            if s not in val_dict:
                val_dict.append(s)

    val_dict.sort()
    val_dict = dict([(v, i) for i, v in enumerate(val_dict)])
    print(val_dict)
    for r in record_list:
        sig, time, rr, counter, ann = get_ls_signal(r)

        inted_sig = []
        for s in sig[:, 0]:
            inted_sig.append(val_dict[s])
        inted_sig = np.array(inted_sig)

    get_graph("101")


if __name__ == "__main__":
    main()

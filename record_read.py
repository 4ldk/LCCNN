import os

import numpy as np
import wfdb
from scipy.io import loadmat

import setting_path as PATH


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


if __name__ == "__main__":
    main()

import os
import random

import numpy as np
import scipy.io
import wfdb

import setting_path as PATH

"""
normal beats (N)
supraventricular ectopic beats (S)
ventricular ectopic beats (V)
fusion beats (F)

N:N
L:N
R:N
V:V
E:V
A:S
S:S
F:F
j:Nodal (junctional) escape beat
J:Nodal (junctional) premature beat
a:Aberrated atrial premature beat
e:Atrial escape beat
Q:Unclassifiable beat
x:blocked APC
|:Isolated QRS-like artifact
+:Rythm change
~:signal quality change
":comment annotation
[:Start of ventricular flutter
!:ventricular flutter wave
]:End of ventricular flutter
"""


def reset_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


reset_seed(123)


class BaseECGDatasetPreprocessor(object):
    def __init__(self, window_size=720):

        self.dataset_root = PATH.ecg_path
        self.dataset_dir = PATH.mit_path
        self.window_size = window_size
        self.sample_rate = 360.0

        self.labels = ["N", "V", "S", "F"]
        self.valid_symbols = [
            "N",
            "L",
            "R",
            "V",
            "E",
            "A",
            "S",
            "F",
            "j",
            "J",
            "a",
            "e",
        ]
        self.label_map = {
            "N": "N",
            "L": "N",
            "R": "N",
            "j": "N",
            "e": "N",
            "V": "V",
            "E": "V",
            "A": "S",
            "S": "S",
            "a": "S",
            "J": "S",
            "F": "F",
        }

    def _load_data(self, base_record, channel=0):

        record_name = os.path.join(self.dataset_dir, str(base_record))
        signals, fields = wfdb.rdsamp(record_name)
        assert fields["fs"] == self.sample_rate
        annotation = wfdb.rdann(record_name, "atr")
        symbols = annotation.symbol
        positions = annotation.sample

        return signals[:, channel], symbols, positions

    def _normalize_signal(self, signal, method="std"):
        if method == "minmax":

            min_val = np.min(signal)
            max_val = np.max(signal)
            return (signal - min_val) / (max_val - min_val)

        elif method == "std":

            signal = (signal - np.mean(signal)) / np.std(signal)
            return signal

        else:
            raise ValueError("Invalid method: {}".format(method))

    def _segment_data(self, signal, symbols, positions):

        X, y = [], []
        sig_len = len(signal)

        for i, s in enumerate(symbols):
            start = positions[i] - self.window_size // 2
            end = positions[i] + self.window_size // 2
            if s in self.valid_symbols and start >= 0 and end <= sig_len:
                segment = signal[start:end]
                assert len(segment) == self.window_size, "Invalid length"
                X.append(segment)
                y.append(self.labels.index(self.label_map[s]))

        return np.array(X), np.array(y)

    def preprocess_dataset(self, save_dir, normalize=True):

        record_list = PATH.record_list
        train_X, train_y, valid_X, valid_y = [], [], [], []

        for r in record_list:

            signal, symbols, positions = self._load_data(r)

            if normalize:
                signal = self._normalize_signal(signal)

            X, y = self._segment_data(signal, symbols, positions)

            shuffled = list(zip(X, y))
            random.shuffle(shuffled)
            X, y = zip(*shuffled)
            data_num = len(X)

            X = np.array(X)
            y = np.array(y)

            train_num = int(data_num * 0.8)

            train_X.append(X[:train_num])
            train_y.append(y[:train_num])
            valid_X.append(X[train_num:])
            valid_y.append(y[train_num:])

        save_dir = os.path.join(self.dataset_root, save_dir, "train")
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X.npy"), np.vstack(train_X))
        np.save(os.path.join(save_dir, "y.npy"), np.concatenate(train_y))

        save_dir = os.path.join(self.dataset_root, save_dir, "valid")
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X.npy"), np.vstack(valid_X))
        np.save(os.path.join(save_dir, "y.npy"), np.concatenate(valid_y))


class LCADCDatasetPreprocessor(object):
    def __init__(self) -> None:
        self.labels = ["N", "V", "S", "F"]
        self.valid_symbols = [
            "N",
            "L",
            "R",
            "V",
            "E",
            "A",
            "S",
            "F",
            "j",
            "J",
            "a",
            "e",
        ]
        self.label_map = {
            "N": "N",
            "L": "N",
            "R": "N",
            "j": "N",
            "e": "N",
            "V": "V",
            "E": "V",
            "A": "S",
            "S": "S",
            "a": "S",
            "J": "S",
            "F": "F",
        }

    def get_lc_signal(self, base_record, channel=0):
        file_name = "Rec" + base_record + "_ED_ch" + str(channel + 1)
        path = os.path.join(PATH.mit_lc_path, file_name)

        data = scipy.io.loadmat(path)["edECG"]
        sig = data[0][0]

        dtype = data.dtype.fields
        signal = sig[0].squeeze().tolist()
        time = sig[1].squeeze().tolist()
        for s, d in zip(sig, dtype.keys()):
            if d == "ann":
                ann_time = s.squeeze().tolist()
            elif d == "anntype":
                ann = s.squeeze().tolist()
            elif d == "counter":
                counter = s.squeeze().tolist()

        return signal, time, counter, ann, ann_time

    def annotation_lc(self, data, window_size=120, time_size=1, without_time=False):

        sig, time, counter, ann, ann_time = data
        X, y, t, c = [], [], [], []
        sig_len = len(sig)

        if without_time:
            dif = (np.array(sig)[1:] == np.array(sig)[:-1]).tolist()
            pop_num = 0
            for i in range(sig_len):
                if i >= sig_len - pop_num:
                    break
                elif dif[i - pop_num]:
                    sig.pop(i - pop_num)
                    time.pop(i - pop_num)
                    dif.pop(i - pop_num)
                    ann_time = [a - 1 if a > i - pop_num else a for a in ann_time]
                    pop_num += 1
            print(sig_len, len(sig))
            sig_len = len(sig)

        for i, a in enumerate(ann):

            start = ann_time[i] - window_size // 2
            end = ann_time[i] + window_size // 2
            if a in self.valid_symbols and start >= 0 and end <= sig_len:
                segment = sig[start:end]
                time_segment = [t - time[start] for t in time[start:end]]
                counter_segment = counter[start:end]
                assert len(segment) == window_size, "Invalid length"

                X.append(segment)
                y.append(self.labels.index(self.label_map[ann[i]]))
                t.append(time_segment)
                c.append(counter_segment)

        X = np.array(X)
        y = np.array(y)
        t = np.array(t)
        c = np.array(c)
        data = [X, y, t, c]

        return data

    def concate_dataset(self, data, save_dir):

        X = np.concatenate([d[0] for d in data])
        y = np.concatenate([d[1] for d in data])
        t = np.concatenate([d[2] for d in data])
        c = np.concatenate([d[3] for d in data])
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X.npy"), X)
        np.save(os.path.join(save_dir, "y.npy"), y)
        np.save(os.path.join(save_dir, "t.npy"), t)
        np.save(os.path.join(save_dir, "c.npy"), c)

    def preprocess_dataset(self, save_dir="lc_preprocessed", without_time=False):

        pathes = PATH.record_list
        save_dir = os.path.join(PATH.ecg_path, save_dir)
        if without_time:
            save_dir = os.path.join(PATH.ecg_path, "without_time")
        train_data, valid_data = [], []

        for p in pathes:

            data = self.get_lc_signal(p)
            data = self.annotation_lc(data, without_time=without_time)
            shuffled = list(zip(*data))
            random.shuffle(shuffled)

            data_num = len(shuffled)
            shuffled = list(zip(*shuffled))
            train_num = int(data_num * 0.8)
            train_data.append([s[:train_num] for s in shuffled])
            valid_data.append([s[train_num:] for s in shuffled])

        train_dir = os.path.join(save_dir, "train")
        self.concate_dataset(train_data, train_dir)
        valid_dir = os.path.join(save_dir, "valid")
        self.concate_dataset(valid_data, valid_dir)


if __name__ == "__main__":

    # Preprocessing normal ADC
    save_dir = "preprocessed"
    BaseECGDatasetPreprocessor(window_size=360).preprocess_dataset(save_dir=save_dir)

    X_train = np.load(os.path.join(PATH.ecg_path, save_dir, "train", "X.npy"))
    y_train = np.load(os.path.join(PATH.ecg_path, save_dir, "train", "y.npy"))

    print("X_train.shape = ", X_train.shape, " \t y_train.shape = ", y_train.shape)

    uniq_train, counts_train = np.unique(y_train, return_counts=True)
    print("y_train count each labels: ", dict(zip(uniq_train, counts_train)))

    # Preprocessing lc ADC
    save_dir = "lc_preprocessed"
    LCADCDatasetPreprocessor().preprocess_dataset(save_dir=save_dir)

    X_train = np.load(os.path.join(PATH.ecg_path, save_dir, "train", "X.npy"))
    y_train = np.load(os.path.join(PATH.ecg_path, save_dir, "train", "y.npy"))

    print("X_train.shape = ", X_train.shape, " \t y_train.shape = ", y_train.shape)

    uniq_train, counts_train = np.unique(y_train, return_counts=True)
    print("y_train count each labels: ", dict(zip(uniq_train, counts_train)))

    # Preprocessing lc ADC w/o max time
    save_dir = "lc_preprocessed"
    LCADCDatasetPreprocessor().preprocess_dataset(save_dir=save_dir, without_time=True)

    X_train = np.load(os.path.join(PATH.ecg_path, "without_time", "train", "X.npy"))
    y_train = np.load(os.path.join(PATH.ecg_path, "without_time", "train", "y.npy"))

    print("X_train.shape = ", X_train.shape, " \t y_train.shape = ", y_train.shape)

    uniq_train, counts_train = np.unique(y_train, return_counts=True)
    print("y_train count each labels: ", dict(zip(uniq_train, counts_train)))

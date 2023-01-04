import os
import warnings

import numpy as np
import pytorch_lightning as pl
from imblearn.over_sampling import SMOTE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

import models
import setting_path as PATH
from datamodules import ECGDataset
from lightning_net import Net


def main():

    warnings.simplefilter("ignore")
    load_dir = "lc_preprocessed"  # preprocessed or lc_preprocessed
    epochs = 1000
    num_gpu = 1
    lr = 1e-3
    batch_size = 512
    dropout = 0.6
    dence_input = 2752  # if window_size = 360→2752 120→192 252→1856 models.pyで調査
    model = models.resnet34(4)

    X_train = np.load(os.path.join(PATH.ecg_path, load_dir, "train", "X.npy"))
    y_train = np.load(os.path.join(PATH.ecg_path, load_dir, "train", "y.npy"))
    X_valid = np.load(os.path.join(PATH.ecg_path, load_dir, "valid", "X.npy"))
    y_valid = np.load(os.path.join(PATH.ecg_path, load_dir, "valid", "y.npy"))

    sm = SMOTE()

    if load_dir == "lc_preprocessed":

        t_train = np.load(os.path.join(PATH.ecg_path, load_dir, "train", "t.npy"))
        t_valid = np.load(os.path.join(PATH.ecg_path, load_dir, "valid", "t.npy"))
        # c_train = np.load(os.path.join(PATH.ecg_path, load_dir, "train", "t.npy")) - 1
        # c_valid = np.load(os.path.join(PATH.ecg_path, load_dir, "valid", "t.npy")) - 1

        X_train = np.stack([X_train, t_train]).transpose(1, 0, 2).reshape(-1, 240)  # c_train or t_train
        X_train, y_train = sm.fit_resample(X_train, y_train)
        X_train = X_train.reshape(-1, 2, 120)

        X_valid = np.stack([X_valid, t_valid]).transpose(1, 0, 2)  # c_valid or t_valid

    else:
        X_train, y_train = sm.fit_resample(X_train, y_train)

    print("X_train.shape = ", X_train.shape, " \t y_train.shape = ", y_train.shape)
    print("X_valid.shape = ", X_valid.shape, " \t y_valid.shape = ", y_valid.shape)

    uniq_train, counts_train = np.unique(y_train, return_counts=True)
    print("y_train count each labels: ", dict(zip(uniq_train, counts_train)))

    uniq_test, counts_test = np.unique(y_valid, return_counts=True)
    print("y_test count each labels: ", dict(zip(uniq_test, counts_test)))

    train_set = ECGDataset(X_train, y_train)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    valid_set = ECGDataset(X_valid, y_valid)
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    print("input shape is")
    for x, y in train_loader:
        print(x.shape, y.shape)
        break

    net = Net(model, lr)

    callbacks = []
    checkpoint = ModelCheckpoint(
        dirpath="./check_point",
        filename="{epoch}-{recall:.2f}",
        monitor="acc",
        save_last=True,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )
    callbacks.append(checkpoint)
    callbacks.append(
        EarlyStopping(
            "recall",
            patience=300,
            verbose=True,
            mode="max",
            check_on_train_epoch_end=False,
        )
    )
    callbacks.append(
        EarlyStopping(
            "loss",
            patience=300,
            verbose=True,
            mode="min",
            check_on_train_epoch_end=False,
        )
    )
    trainer = pl.Trainer(max_epochs=epochs, gpus=num_gpu, accelerator="gpu", check_val_every_n_epoch=10)
    trainer.fit(net, train_loader, valid_loader)
    trainer.test(dataloaders=valid_loader, ckpt_path="best")


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()

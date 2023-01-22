import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from torch import nn, optim

from utils import recall_score


class ECGDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].astype(np.float32), self.y[idx]


class Net(pl.LightningModule):
    def __init__(self, input_layer, model, lr):
        super().__init__()

        print("Making Model...")
        self.input = input_layer
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            amsgrad=True,
        )
        self.cm = np.zeros([4, 4])

    def forward(self, input):
        x = self.input(input)
        output = self.model(x)
        return output

    def loss_fn(self, pred, label):

        pred = pred.reshape(pred.shape[0], -1)
        label = label.view(-1).to(torch.int64)
        loss = self.criterion(pred, label)

        return loss

    def training_step(self, batch, batch_idx):

        input, label = batch

        pred = self.forward(input)
        loss = self.loss_fn(pred, label)
        self.log("loss", loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):

        input, label = batch
        batch_size = input.shape[0]
        label = label.to("cpu")
        pred = self.forward(input)
        pred = torch.argmax(pred, dim=1).to("cpu")

        pred = pred.squeeze(-1)
        acc = (pred == label).sum().item() / batch_size
        pred = pred.view(-1)
        label = label.view(-1)
        recall = recall_score(label, pred, average="macro").tolist()

        return {"acc": acc, "recall": recall}

    def validation_epoch_end(self, outputs):
        ave_acc = torch.tensor([x["acc"] for x in outputs]).to(torch.float).mean()
        ave_recall = torch.tensor([x["recall"] for x in outputs]).to(torch.float).mean()

        self.log("acc", ave_acc)
        self.log("recall", ave_recall)
        self.log("lr", self.optimizer.param_groups[0]["lr"])

        return {"acc": ave_acc}

    def test_step(self, batch, batch_idx):

        input, label = batch
        batch_size = input.shape[0]
        label = label.to("cpu")
        pred = self.forward(input)
        pred = torch.argmax(pred, dim=1).to("cpu")

        pred = pred.squeeze(-1)
        acc = (pred == label).sum().item() / batch_size
        pred = pred.view(-1)
        label = label.view(-1)
        recall = recall_score(label, pred, average="macro").tolist()

        label_name = list(range(4))
        self.cm += confusion_matrix(label, pred, labels=label_name)

        self.log("test_acc", acc)
        self.log("test_recall", recall)

    def test_epoch_end(self, outputs) -> None:

        cm = self.cm

        label_name = ["N", "V", "S", "F"]
        columns_labels = ["pred_" + l_n for l_n in label_name]
        index_labels = ["act_" + l_n for l_n in label_name]
        cm = pd.DataFrame(cm, columns=columns_labels, index=index_labels)
        print(cm.to_markdown())

        return super().test_epoch_end(outputs)

    def configure_optimizers(self):
        return [self.optimizer]

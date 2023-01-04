import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

warnings.simplefilter("ignore")

from mmcv.cnn.utils import flops_counter


class LSTM(nn.Module):
    def __init__(self, batch_size, dim, hidden_dim, pred_num, num_lstm, device="cuda") -> None:
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_lstm
        self.batch_size = batch_size
        self.linear1 = nn.Linear(1, self.dim)
        # self.lstm = nn.LSTM(self.dim, self.hidden_dim, num_layers=self.num_layer, batch_first=True, dropout=0.3, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=4 * dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layer)
        self.linear2 = nn.Linear(self.dim * 720, pred_num)

        self.h_0 = Variable(torch.zeros(self.num_layer * 2, self.batch_size, self.hidden_dim).to(device))
        self.c_0 = Variable(torch.zeros(self.num_layer * 2, self.batch_size, self.hidden_dim).to(device))

        pos = torch.arange(720).to(dtype=torch.long, device=device)
        self.positions = pos.unsqueeze(0)
        self.position_embeddings = nn.Embedding(720, dim)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class CNN(nn.Module):
    def __init__(self, dense_input, dropout=0.5) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=3, bias=False)
        self.conv2 = nn.Conv1d(64, self.in_channels, kernel_size=3, bias=False)
        self.conv3 = nn.Conv1d(64, self.in_channels, kernel_size=3, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dence = nn.Linear(dense_input, 128)
        self.out = nn.Linear(128, 4)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dence(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x


class LCCNN(nn.Module):
    def __init__(self, dense_input, dropout=0.5) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(2, self.in_channels, kernel_size=3, bias=False)
        self.conv2 = nn.Conv1d(64, self.in_channels, kernel_size=3, bias=False)
        self.conv3 = nn.Conv1d(64, self.in_channels, kernel_size=3, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dence = nn.Linear(dense_input, 128)
        self.out = nn.Linear(128, 4)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dence(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x


class LCCNNLight(nn.Module):
    def __init__(self, dropout=0.5) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(2, self.in_channels, kernel_size=3, bias=False)
        self.conv2 = nn.Conv1d(64, self.in_channels, kernel_size=3, bias=False)
        self.conv3 = nn.Conv1d(64, self.in_channels, kernel_size=3, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lastpool = nn.MaxPool1d(kernel_size=26, stride=1)
        self.dence = nn.Linear(64, 128)
        self.out = nn.Linear(128, 4)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lastpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.dence(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x


class LCCNNLight2(nn.Module):
    def __init__(self, dropout=0.5) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(2, self.in_channels, kernel_size=3, bias=False)
        self.conv2 = nn.Conv1d(64, self.in_channels, kernel_size=3, bias=False)
        self.conv3 = nn.Conv1d(64, self.in_channels, kernel_size=3, bias=False)
        self.conv4 = nn.Conv1d(64, self.in_channels, kernel_size=13, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dence = nn.Linear(64, 128)
        self.out = nn.Linear(128, 4)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)

        x = self.dence(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x


class TimeEmbeddingCNN(nn.Module):
    def __init__(self, dense_input, dropout=0.5) -> None:
        super().__init__()
        self.in_channels = 64
        self.time_embedding = nn.Embedding(64, 64)
        self.linear = nn.Linear(1, 64)
        self.conv2 = nn.Conv1d(64, self.in_channels, kernel_size=3, bias=False)
        self.conv3 = nn.Conv1d(64, self.in_channels, kernel_size=3, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dence = nn.Linear(dense_input, 128)
        self.out = nn.Linear(128, 4)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        c = input[:, 1].to(torch.int)
        x = input[:, 0].unsqueeze(-1)
        x = self.linear(x)
        c = self.time_embedding(c)
        x = x + c
        x = x.transpose(2, 1)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dence(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x


class TimeSinCNN(nn.Module):
    def __init__(self, dense_input, dropout=0.5, batch=512, length=120, dim=64) -> None:
        super().__init__()
        self.in_channels = 64

        self.pe = torch.zeros(batch, length, dim)
        pos = torch.arange(0, dim, 2)
        self.div_term = torch.pow((1 / 10000), pos / dim)
        self.linear = nn.Linear(1, 64)
        self.conv2 = nn.Conv1d(64, self.in_channels, kernel_size=3, bias=False)
        self.conv3 = nn.Conv1d(64, self.in_channels, kernel_size=3, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dence = nn.Linear(dense_input, 128)
        self.out = nn.Linear(128, 4)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        c = input[:, 1].to(torch.int).unsqueeze(2)
        x = input[:, 0].unsqueeze(-1)
        x = self.linear(x)

        pe = self.pe.to(c.device)
        div_term = self.div_term.to(c.device)
        pe[:, :, 0::2] = torch.sin(c * div_term)
        pe[:, :, 1::2] = torch.cos(c * div_term)
        tim_emb = Variable(pe, requires_grad=False)

        x = x + tim_emb
        x = x.transpose(2, 1)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dence(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x


# 以下ResNetの実装 参考文献(ほぼコピペコード): https://pystyle.info/pytorch-resnet/


def conv3(in_channels, out_channels, stride=1):
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = conv3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)

        if in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv1(in_channels, channels * self.expansion, stride),
                nn.BatchNorm1d(channels * self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(x)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = conv1(in_channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = conv3(channels, channels, stride)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv3 = conv1(channels, channels * self.expansion)
        self.bn3 = nn.BatchNorm1d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv1(in_channels, channels * self.expansion, stride),
                nn.BatchNorm1d(channels * self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv1d(2, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride):
        layers = []

        layers.append(block(self.in_channels, channels, stride))

        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet34(num_classes=1):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


if __name__ == "__main__":

    m = CNN(1792)

    input_shape = (2, 120)
    t = flops_counter.get_model_complexity_info(m, input_shape, as_strings=False)
    print(t)

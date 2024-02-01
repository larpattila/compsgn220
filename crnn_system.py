#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from torch.nn import Module, Sequential, Dropout2d, GRU, Linear, Conv2d, MaxPool2d, BatchNorm2d, ReLU


class MyCRNNSystem(Module):
    def __init__(self) -> None:
        super(MyCRNNSystem, self).__init__()

        self.cnn_block_1 = Sequential(
            Conv2d(in_channels=1,
                   out_channels=?,
                   kernel_size=(?, ?),
                   stride=(?, ?),
                   padding=(?, ?)),
            ReLU(),
            BatchNorm2d(?),
            MaxPool2d(kernel_size=?,
                      stride=?),
            Dropout2d(0.5))

        self.cnn_block_2 = Sequential(
            Conv2d(in_channels=?,
                   out_channels=?,
                   kernel_size=(?, ?),
                   stride=(?, ?),
                   padding=(?, ?)),
            ReLU(),
            BatchNorm2d(?),
            MaxPool2d(kernel_size=(?, ?),
                      stride=(?, ?)),
            Dropout2d(0.5))

        self.rnn_layer = GRU(input_size=?,
                             hidden_size=?,
                             num_layers=?,
                             batch_first=True)

        self.linear = Linear(in_features=?, out_features=?)

    def forward(self,
                X: Tensor) -> Tensor:
        X = X.float()
        X = X if X.ndimension() == 4 else X.unsqueeze(1)
        # apply cnn_block_1 to X
        cnn_out_1 = ?
        # apply cnn_block_2 to cnn_out_1
        cnn_out_2 = ?
        # apply permute 
        cnn_out_2 = ?
        # reshape
        cnn_out_2 = ?
        # apply rnn_layer 
        rnn_out, _ = ?
        # apply linear layer
        y_hat = ?

        return y_hat

# EOF

# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device = device
        self.model = torch.nn.LSTM(input_size=vocabulary_size, hidden_size=lstm_num_hidden,
                                   num_layers=lstm_num_layers, batch_first=True)#.to(device)
        self.output_layer = torch.nn.Linear(lstm_num_hidden, vocabulary_size)#, device=device)

    def forward(self, x, h_0=None, c_0=None):
        if h_0 is not None and c_0 is not None:
            output, (h_n, c_n) = self.model(x, (h_0, c_0))
        else:
            output, (h_n, c_n) = self.model(x)
        output = self.output_layer(output)
        return output, (h_n, c_n)

    def forward_autoregressive(self, x, h_0=None, c_0=None):
        outputs = torch.empty(x.shape[0], self.seq_length, x.shape[-1], device=self.device)

        if h_0 is not None and c_0 is not None:
            output, (h_n, c_n) = self.model(x, (h_0, c_0))
        else:
            output, (h_n, c_n) = self.model(x)
        output = self.output_layer(output)

        aux = torch.eye(self.vocabulary_size, device=self.device)
        output = aux[output.argmax(-1)]
        outputs[:, 0, :] = output
        for i in range(self.seq_length-1):
            output, (h_n, c_n) = self.model(output, (h_n, c_n))
            output = self.output_layer(output)
            output = aux[output.argmax(-1)]
            outputs[:, i, :] = output
        return outputs, (h_n, c_n)


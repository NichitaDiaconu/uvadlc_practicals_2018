################################################################################
# MIT License
#
# Copyright (c) 2018
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

import math
import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu', predict_half=False):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        self.predict_half = predict_half

        self.h = torch.empty(self.batch_size, self.num_hidden, device=self.device)
        self.w_hx = torch.nn.Parameter(torch.empty(self.num_hidden, self.input_dim, device=self.device))
        self.w_hh = torch.nn.Parameter(torch.empty(self.num_hidden, self.num_hidden, device=self.device))
        self.w_ph = torch.nn.Parameter(torch.empty(self.num_classes, self.num_hidden, device=self.device))
        self.b_h = torch.nn.Parameter(torch.zeros(self.num_hidden, device=self.device))
        self.b_p = torch.nn.Parameter(torch.zeros(self.num_classes, device=self.device))
        self._init_parameters()

    def forward(self, x):
        self._reset_state()

        if self.predict_half:
            output = torch.empty(self.batch_size, self.seq_length, self.num_classes, device=self.device)
            for step in range(self.seq_length):
                self._step(x[:, [step]])
                pred = self.h @ self.w_ph.t() + self.b_p
                output[:, step] = pred
        else:
            for step in range(self.seq_length):
                self._step(x[:, [step]])
            output = self.h @ self.w_ph.t() + self.b_p
        #output = torch.nn.Softmax(dim=0)(output)
        return output

    def _step(self, x):
        self.h = torch.nn.Tanh()(x @ self.w_hx.t() + self.h @ self.w_hh.t() + self.b_h)

    def _init_parameters(self):
        self.h.data.zero_()
        std = math.sqrt(2/(self.w_hx.size(0)+self.w_hx.size(1)))
        self.w_hx.data.normal_(0, std)
        #a = math.sqrt(3.0) * std
        #self.w_hx.data.uniform_(-a, a)
        std = math.sqrt(2 / (self.w_hh.size(0) + self.w_hh.size(1)))
        self.w_hh.data.normal_(0, std)
        #a = math.sqrt(3.0) * std
        #self.w_hh.data.uniform_(-a, a)
        std = math.sqrt(2 / (self.w_ph.size(0) + self.w_ph.size(1)))
        self.w_ph.data.normal_(0, std)
        #a = math.sqrt(3.0) * std
        #self.w_ph.data.uniform_(-a, a)

    def _reset_state(self):
        self.h = torch.empty(self.batch_size, self.num_hidden, device=self.device)

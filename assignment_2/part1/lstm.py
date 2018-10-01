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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu', predict_half=False):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        self.predict_half = predict_half

        self.w_gx = torch.nn.Parameter(torch.empty(self.num_hidden, self.input_dim, device=self.device))
        self.w_ix = torch.nn.Parameter(torch.empty(self.num_hidden, self.input_dim, device=self.device))
        self.w_fx = torch.nn.Parameter(torch.empty(self.num_hidden, self.input_dim, device=self.device))
        self.w_ox = torch.nn.Parameter(torch.empty(self.num_hidden, self.input_dim, device=self.device))

        self.w_gh = torch.nn.Parameter(torch.empty(self.num_hidden, self.num_hidden, device=self.device))
        self.w_ih = torch.nn.Parameter(torch.empty(self.num_hidden, self.num_hidden, device=self.device))
        self.w_fh = torch.nn.Parameter(torch.empty(self.num_hidden, self.num_hidden, device=self.device))
        self.w_oh = torch.nn.Parameter(torch.empty(self.num_hidden, self.num_hidden, device=self.device))

        self.w_ph = torch.nn.Parameter(torch.empty(self.num_classes, self.num_hidden, device=self.device))

        self.b_g = torch.nn.Parameter(torch.empty(self.num_hidden, device=self.device))
        self.b_i = torch.nn.Parameter(torch.empty(self.num_hidden, device=self.device))
        self.b_f = torch.nn.Parameter(torch.empty(self.num_hidden, device=self.device))
        self.b_o = torch.nn.Parameter(torch.empty(self.num_hidden, device=self.device))

        self.b_p = torch.nn.Parameter(torch.empty(self.num_classes, device=self.device))

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
        g = torch.nn.Tanh()(x @ self.w_gx.t() + self.h @ self.w_gh.t() + self.b_g)
        i = torch.nn.Sigmoid()(x @ self.w_ix.t() + self.h @ self.w_ih.t() + self.b_i)
        f = torch.nn.Sigmoid()(x @ self.w_fx.t() + self.h @ self.w_fh.t() + self.b_f)
        o = torch.nn.Sigmoid()(x @ self.w_ox.t() + self.h @ self.w_oh.t() + self.b_o)
        self.c = g * i + self.c * f

        self.h = torch.nn.Tanh()(self.c) * o

    def _init_parameters(self):
        torch.nn.init.xavier_normal_(self.w_gx)
        # std = math.sqrt(2 / (self.w_gx.size(0) + self.w_gx.size(1)))
        # self.w_gx.data.normal_(0, std)

        torch.nn.init.xavier_normal_(self.w_gh)
        # std = math.sqrt(2 / (self.w_gh.size(0) + self.w_gh.size(1)))
        # self.w_gh.data.normal_(0, std)

        torch.nn.init.xavier_normal_(self.w_ix)
        # std = math.sqrt(2 / (self.w_ix.size(0) + self.w_ix.size(1)))
        # self.w_ix.data.normal_(0, std)

        torch.nn.init.xavier_normal_(self.w_ih)
        # std = math.sqrt(2 / (self.w_ih.size(0) + self.w_ih.size(1)))
        # self.w_ih.data.normal_(0, std)

        torch.nn.init.xavier_normal_(self.w_fx)
        # std = math.sqrt(2 / (self.w_fx.size(0) + self.w_fx.size(1)))
        # self.w_fx.data.normal_(0, std)

        torch.nn.init.xavier_normal_(self.w_fh)
        # std = math.sqrt(2 / (self.w_fh.size(0) + self.w_fh.size(1)))
        # self.w_fh.data.normal_(0, std)

        torch.nn.init.xavier_normal_(self.w_ox)
        # std = math.sqrt(2 / (self.w_ox.size(0) + self.w_ox.size(1)))
        # self.w_ox.data.normal_(0, std)

        torch.nn.init.xavier_normal_(self.w_oh)
        # std = math.sqrt(2 / (self.w_oh.size(0) + self.w_oh.size(1)))
        # self.w_oh.data.normal_(0, std)

        torch.nn.init.xavier_uniform_(self.w_ph)
        # std = math.sqrt(2 / (self.w_ph.size(0) + self.w_ph.size(1)))
        # self.w_ph.data.normal_(0, std)

        self.b_g.data.zero_()
        self.b_i.data.zero_()
        self.b_f.data.zero_()
        self.b_o.data.zero_()
        self.b_p.data.zero_()

    def _reset_state(self):
        self.h = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        self.c = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
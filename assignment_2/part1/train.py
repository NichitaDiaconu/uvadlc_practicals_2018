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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM
import os
import pickle

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def config_to_str(config):
    name = ''
    name += 'model_type=' + str(config.model_type)+', '
    name += 'input_length=' + str(config.input_length) + ', '
    name += 'learning_rate='+str(config.learning_rate)
    return name

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    model_dir = config.summary_path + config_to_str(config) + '/'
    os.makedirs(model_dir)  # , exist_ok=True)

    # add assets to filename if we removoed it

    with open(model_dir + 'config.pkl', 'wb+') as fd:
        pickle.dump(config, fd)

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden,
                           config.num_classes, config.batch_size, config.device, config.predict_half)
    else:
        model = LSTM(config.input_length, config.input_dim, config.num_hidden,
                     config.num_classes, config.batch_size, config.device, config.predict_half)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1, config.batch_size, config.train_steps)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    losses = []
    accuracies = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()
        batch_inputs, batch_targets = Variable(batch_inputs.to(device)), Variable(batch_targets.to(device))

        # Add more code here ...
        optimizer.zero_grad()
        batch_output = model.forward(batch_inputs)
        if config.predict_half:
            #MANY
            long_target = torch.cat((batch_inputs[:, int(config.input_length/2)+1:], batch_targets.unsqueeze(-1).float()),1)
            long_predictions = batch_output[:, int(config.input_length/2):]
            loss = criterion(long_predictions.contiguous().view(-1, long_predictions.size()[-1]),
                      long_target.contiguous().view(-1).long())
            accuracy = float(torch.sum(long_predictions.argmax(2) == long_target.long())) / (long_predictions.shape[0] * long_predictions.shape[1])
        else:
            #ONE
            predictions = batch_output
            loss = criterion(predictions, batch_targets)
            accuracy = float(torch.sum(predictions.argmax(1) == batch_targets)) / predictions.shape[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:
            losses.append(loss.item())
            accuracies.append(accuracy)
            if step % 200 == 0:
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

    with open(model_dir + 'learning_curves.pkl', 'wb+') as fd:
        pickle.dump((losses, accuracies), fd)
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--model_type', type=str, default="LSTM", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=2000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--predict_half', type=bool, default=False, help="boolean for predicting all the sequence or just the last element")

    config = parser.parse_args()

    # Train the model

for model_type in ['LSTM']:
    config.model_type = model_type
    for input_length in [30, 65, 95]:
        config.input_length = input_length
        train(config)
        # model_dir = config.summary_path + config_to_str(config) + '/'
        # with open(model_dir + 'learning_curves.pkl', 'rb') as fd:
        #     a = pickle.load(fd)
        # print("a")
#train(config)

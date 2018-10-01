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

import os
import time
from datetime import datetime
import argparse

import numpy as np
import pickle

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from part3.dataset import TextDataset
from part3.model import TextGenerationModel

################################################################################
def config_to_str(config):
    name = ''
    name += 'seq_length='+str(config.seq_length)+', '
    name += 'txt_file='+str(config.txt_file)+', '
    name += 'learning_rate='+str(config.learning_rate)+', '
    name += 'lstm_num_hidden='+str(config.lstm_num_hidden)+', '
    name += 'lstm_num_layers='+str(config.lstm_num_layers)
    return name


def train(config):

    # Initialize the device which to run the model on
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize the dataset and data loader (note the +1)
    #MAKE A DIR FOR EVERY MODEL
    model_dir = config.summary_path+config_to_str(config)+'/'
    os.makedirs(model_dir)#, exist_ok=True)

    #add assets to filename if we removoed it
    if 'assets' not in config.txt_file:
        config.txt_file = './assets/' + config.txt_file

    with open(model_dir+'config.pkl', 'wb+') as fd:
        pickle.dump(config, fd)

    dataset = TextDataset(config.txt_file, config.seq_length, config.batch_size, int(config.train_steps))
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
                                config.lstm_num_hidden, config.lstm_num_layers, device)
    model.to(device)
    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters())

    losses = []
    accuracies = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################
        # Add more code here ...
        #######################################################
        ###to one hot
        batch_inputs = (torch.stack(batch_inputs).t())
        aux = torch.eye(dataset.vocab_size, device=device)
        batch_inputs = Variable(aux[batch_inputs])
        ###
        batch_targets = Variable(torch.stack(batch_targets).t().to(device))
        optimizer.zero_grad()
        output, (h_n, c_n) = model.forward(batch_inputs)
        loss = criterion(output.view(-1, output.shape[-1]), batch_targets.view(-1))
        accuracy = float(torch.sum(output.argmax(2) == batch_targets)) / (output.shape[0] * output.shape[1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    int(config.train_steps), config.batch_size, examples_per_second,
                    accuracy, loss
            ))
            accuracies.append(accuracy)
            losses.append(loss.item())

        if config.sample_every != 0 and step % config.sample_every == 0:
            #GENERATE 30
            rand_idx = np.random.randint(0, dataset.vocab_size)
            aux = torch.eye(dataset.vocab_size, device=device)
            batch_inputs_gen = Variable(aux[rand_idx].unsqueeze(0).unsqueeze(0))

            output_30, _ = model.forward_autoregressive(batch_inputs_gen)

            one_hot_sentence = torch.cat((batch_inputs_gen, output_30), dim=1)
            sentence_30 = dataset.convert_to_string(one_hot_sentence.argmax(dim=-1)[0].cpu().numpy().tolist())
            # with open(model_dir + 'gen_30.pkl', 'wb+') as fd:
            #     pickle.dump(sentence_30, fd)
            print("HERE")
            print(sentence_30)

            #GENERATE 30 AFTER 30. BONUS
            sentence_generate_30_30 = output[-1].unsqueeze(0)
            last_char_idx = sentence_generate_30_30[0, -1].argmax()
            batch_inputs_gen = Variable(aux[last_char_idx].unsqueeze(0).unsqueeze(0))
            output_30_30, _ = model.forward_autoregressive(batch_inputs_gen, h_n[:,-1,:].unsqueeze(1).contiguous(), c_n[:,-1,:].unsqueeze(1).contiguous())

            one_hot_sentence = torch.cat((sentence_generate_30_30, output_30_30), dim=1)
            sentence_generate_30_30 = dataset.convert_to_string(

            one_hot_sentence.argmax(dim=-1)[0].cpu().numpy().tolist())
            one_hot_sentence = torch.cat((batch_inputs[-1].unsqueeze(0), output_30_30), dim=1)
            sentence_truth_30_30 = dataset.convert_to_string(one_hot_sentence.argmax(dim=-1)[0].cpu().numpy().tolist())

            print(sentence_truth_30_30)
            print(sentence_generate_30_30)

            # with open(model_dir + 'gen_30_30.pkl', 'wb+') as fd:
            #     pickle.dump((sentence_generate_30_30, sentence_truth_30_30), fd)

        # if step == config.train_steps:
        #     # If you receive a PyTorch data-loader error, check this bug report:
        #     # https://github.com/pytorch/pytorch/pull/9655
        #     break
    with open(model_dir+'learning_curves.pkl', 'wb+') as fd:
        pickle.dump((losses, accuracies), fd)
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='book_NL_darwin_reis_om_de_wereld.txt', help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6+2, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=200000, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    # train(config)
    with open('./summaries/seq_length=30, txt_file=book_NL_darwin_reis_om_de_wereld.txt, learning_rate=0.002, lstm_num_hidden=128, lstm_num_layers=2/learning_curves.pkl', 'rb') as fd:
        config = pickle.load(fd)
    print(2)


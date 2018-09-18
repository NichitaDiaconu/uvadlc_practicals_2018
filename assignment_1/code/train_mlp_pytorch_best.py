"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

import torch
from torch import nn
from torch.autograd import Variable

from mlp_pytorch_best import MLP
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '2000,1000,200,200,200,200,200,200,200'
LEARNING_RATE_DEFAULT = 1e-4
MAX_STEPS_DEFAULT = 100000
BATCH_SIZE_DEFAULT = 300
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  accuracy = torch.sum(targets[np.arange(0, targets.shape[0]), torch.argmax(predictions, dim=1)] == 1)
  accuracy = float(accuracy) / targets.shape[0]
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  dataset_dict = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
  train_loader = dataset_dict['train']
  test_loader = dataset_dict['test']
  test_images = Variable(torch.tensor(test_loader.images.reshape(test_loader.images.shape[0], -1)))
  test_labels = torch.tensor(test_loader.labels)
  model = MLP(n_inputs=32 * 32 * 3, n_hidden=dnn_hidden_units, n_classes=10).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
  criterion = nn.CrossEntropyLoss()

  test_accs = []
  train_accs = []
  losses = []
  for epoch in range(FLAGS.max_steps):
    model.train()
    batch_x, batch_y = train_loader.next_batch(FLAGS.batch_size)
    batch_x, batch_y = Variable(torch.tensor(batch_x).to(device)), Variable(torch.tensor(batch_y).to(device))
    optimizer.zero_grad()
    out = model.forward(batch_x.reshape(FLAGS.batch_size, -1))
    loss = criterion(out, batch_y.max(1)[1])

    losses.append(round(float(loss), 3))
    loss.backward()
    optimizer.step()

    if epoch % FLAGS.eval_freq == 0:
      model.eval()
      # print accuracy on test and train set
      train_acc = accuracy(out, batch_y)
      out = model.forward(test_images.to(device))
      test_acc = accuracy(out, test_labels.to(device))
      print(
        'Train Epoch: {}/{}\tLoss: {:.6f}\tTrain accuracy: {:.6f}\tTest accuracy: {:.6f}'.format(
          epoch, FLAGS.max_steps, loss, train_acc, test_acc))
      test_accs.append(float(test_acc))
      train_accs.append(float(train_acc))

  out = model.forward(test_images.to(device))
  test_acc = accuracy(out, test_labels.to(device))
  print('FINAL Test accuracy: {:.6f}'.format(test_acc))
  print('MAX accuracy: {:.6f}'.format(max(test_accs)))

  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot([i for i in range(0, epoch+1, EVAL_FREQ_DEFAULT)], train_accs)
  plt.plot([i for i in range(0, epoch+1, EVAL_FREQ_DEFAULT)], test_accs)
  plt.legend(["train", "test"])
  plt.ylabel("accuracy")
  plt.xlabel("epoch")
  plt.savefig("mlp_pytorch_accuracy")
  plt.figure()
  plt.plot([i for i in range(0, epoch+1, 1)], losses)
  plt.legend(["loss"])
  plt.ylabel("loss")
  plt.xlabel("epoch")
  plt.savefig("mlp_pytorch_loss")
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  main()

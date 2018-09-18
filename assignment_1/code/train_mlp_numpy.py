"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

import modules
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
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
  accuracy = np.sum(targets[np.arange(0,targets.shape[0]), np.argmax(predictions, axis=1)] == 1)
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
  # return {'train': train, 'validation': validation, 'test': test}
  dataset_dict = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
  train_loader = dataset_dict['train']
  test_loader = dataset_dict['test']

  model = MLP(n_inputs=32*32*3, n_hidden=dnn_hidden_units, n_classes=10)

  test_accs = []
  train_accs = []
  losses = []
  for epoch in range(FLAGS.max_steps):
    batch_x, batch_y = train_loader.next_batch(FLAGS.batch_size)
    out = model.forward(batch_x.reshape(FLAGS.batch_size, -1))

    cross_ent = CrossEntropyModule()
    loss = cross_ent.forward(out, batch_y)
    losses.append(round(loss,3))
    dout = cross_ent.backward(out, batch_y)
    model.backward(dout)

    for layer in model.layers:
      if type(layer) == modules.LinearModule:
        layer.params['weight'] = layer.params['weight'] - FLAGS.learning_rate*layer.grads['weight']
        layer.params['bias'] = layer.params['bias'] - FLAGS.learning_rate*layer.grads['bias']
    if epoch % FLAGS.eval_freq == 0:
      #print accuracy on test and train set
      train_acc = accuracy(out, batch_y)
      out = model.forward(test_loader.images.reshape(test_loader.images.shape[0],-1))
      test_acc = accuracy(out, test_loader.labels)
      print(
        'Train Epoch: {}/{}\tLoss: {:.6f}\tTrain accuracy: {:.6f}\tTest accuracy: {:.6f}'.format(
          epoch, FLAGS.max_steps, loss, train_acc, test_acc))
      test_accs.append(test_acc)
      train_accs.append(train_acc)
  out = model.forward(test_loader.images.reshape(test_loader.images.shape[0], -1))
  test_acc = accuracy(out, test_loader.labels)
  print('FINAL Test accuracy: {:.6f}'.format(test_acc))

  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot([i for i in range(0, MAX_STEPS_DEFAULT, EVAL_FREQ_DEFAULT)], train_accs)
  plt.plot([i for i in range(0, MAX_STEPS_DEFAULT, EVAL_FREQ_DEFAULT)], test_accs)
  plt.legend(["train", "test"])
  plt.ylabel("accuracy")
  plt.xlabel("epoch")
  plt.savefig("accuracy")
  plt.figure()
  plt.plot([i for i in range(0, MAX_STEPS_DEFAULT, 1)], losses)
  plt.legend(["loss"])
  plt.ylabel("loss")
  plt.xlabel("epoch")
  plt.savefig("loss")
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
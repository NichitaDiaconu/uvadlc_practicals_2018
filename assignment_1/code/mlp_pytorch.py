"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(MLP, self).__init__()
    self.n_inputs = n_inputs
    self.n_classes = n_classes
    # Define a list of layers
    self.layers = nn.ModuleList()
    in_sz = n_inputs
    # Add hidden layers
    for sz in n_hidden:
      layer = nn.Linear(in_sz, sz)
      layer.bias.data.fill_(value=0)
      layer.weight.data.normal_(mean=0, std=1e-4)
      self.layers.append(layer)
      in_sz = sz
    # Add final layer
    layer = nn.Linear(in_sz, n_classes)
    layer.bias.data.fill_(value=0)
    layer.weight.data.normal_(mean=0, std=1e-4)
    self.layers.append(layer)
    # Add final layer activation
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    for layer in self.layers[:-1]:
      x = nn.ReLU()(layer(x))
    out = nn.LogSoftmax(-1)(self.layers[-1](x))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

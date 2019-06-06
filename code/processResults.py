import sys
import os
import numpy as np
import glob
from numpy.random import shuffle
from numpy import genfromtxt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
import scipy.stats

import scipy as sp
import scipy.stats
import os
import math
import ast
import sys
import argparse
import configparser
from args import ArgReader

import netBuilder
import torch.nn.functional as F
import torch
import dataLoader
import torch.nn.functional as F
import vis
import trainVal
from sklearn.manifold import TSNE
import netBuilder

#Main functions

class Bunch(object):
  '''Convert a dictionnary into a namespace object'''
  def __init__(self, dict):
    self.__dict__.update(dict)

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    #Reading the comand line arg
    argreader.getRemainingArgs()
    #Getting the args from command line and config file
    args = argreader.args

    args.cuda = not args.no_cuda and torch.cuda.is_available()


if __name__ == '__main__':
    main()

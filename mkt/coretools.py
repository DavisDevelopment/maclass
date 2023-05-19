from mkt.olbi import printing

import itertools
from functools import reduce, wraps
import pickle
from time import sleep
from pprint import pprint
# from numpy import asanyarray, ndarray
import termcolor
# from torch import BoolTensor, random, renorm
# from torch.jit._script import ScriptModule, ScriptFunction
# from mkt.datatools import P, ohlc_resample, pl_binary_labeling, quantize, renormalize, unpack
# from mkt.datatools import get_cache, calc_Xy, load_ts_dump, rescale, norm_batches
   
# from nn.namlp import NacMlp
from cytoolz import *
from cytoolz import itertoolz as iters
import pandas as pd

# from nn.data.core import TwoPaneWindow, TensorBuffer
from pandas import DataFrame
from typing import *
# from nn.ts.classification.fcn_baseline import FCNBaseline2D, FCNNaccBaseline
# from nn.arch.transformer.TransformerDataset import TransformerDataset, generate_square_subsequent_mask
# from mkt.tools import Struct, dotget, gets, maxby, minby, unzip, safe_div, argminby, argmaxby
# from nn.arch.lstm_vae import *
import torch.nn.functional as F
# from nn.ts.classification import LinearBaseline, FCNBaseline, InceptionModel, ResNetBaseline
# from nn.arch.transformer.TransformerDataset import generate_square_subsequent_mask

from sklearn.preprocessing import MinMaxScaler
# from ttools.thunk import Thunk, thunkv, thunk



def shuffle_tensors_in_unison(*all, axis:int=0):
   state = torch.random.get_rng_state()
   # if axis == 0:
   #    return tuple(a[a.size()[axis]]for a in all:
         
   inputs = list(all)
   if len(inputs) == 0:
      return []
   input_ndim = inputs[0].ndim
   input_shape = inputs[0].size()
   
   if abs(axis) > input_ndim-1:
      raise IndexError('invalid axis')
   
   accessor = []
   for i in range(0, axis):
      accessor.append(slice(None))
   results = []
   for i, a in enumerate(inputs):
      torch.random.set_rng_state(state)
      results.append(a[(*accessor, torch.randperm(a.size()[axis]))])
   return tuple(results)
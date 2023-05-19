
from typing import Iterable, Tuple
import pandas as pd
import numpy as np
from torch import Tensor, tensor
import torch
# from nn.data.sampler import DataFrameSampler
from operator import methodcaller
from functools import partial
from itertools import zip_longest
from fn import F
from mkt.tools import nor, unzip, nn, isiterable
from typing import *

class Agent:
   # signal_generators:Dict[str, SignalGenerator]
   
   def __init__(self):
      self.signal_generators = {}
   
   

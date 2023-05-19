
import torch
from torch import Tensor
import numpy as np

from tsai.models.MINIROCKET_Pytorch import MiniRocket, MiniRocketFeatures, MiniRocketHead, get_minirocket_features
from mkt.tools import nn as notnone

class RocketShipOptions:
   def __init__(self, **kwargs):
      _predefined = set(dir(self))
      
      self.num_flat_features = 10_000
      self.label_boundaries = np.arange(-0.5, 0.75, 0.015)
      self.num_input_columns = None
      self.num_input_steps = None
      self.num_classes = len(self.label_boundaries)
      
      mine = (set(dir(self)) - _predefined)
      for k, v in kwargs.items():
         if k in mine:
            setattr(self, k, v)
      
   def datafit(self, X, y):
      self.num_input_columns = X.shape[2]
      self.num_input_steps = X.shape[1]
      if self.num_classes is not None:
         self.num_classes = len(np.unique(y))
      return self

class RocketShip:
   def __init__(self, features=None, head=None, **kwargs):
      o = kwargs.pop('options', None)
      if o is None:
         o = RocketShipOptions(**kwargs)
      self.options = o
      self._initialized = False
      
      self._flat_features = features
      self._flat_features_initialized = False
      
      self._feature_label = head
      
   def init(self):
      if self._flat_features is None:
         assert notnone(self.options.num_input_columns), 'num_input_columns must be specified'
         assert notnone(self.options.num_input_steps), 'num_input_steps must be specified'
         assert notnone(self.options.num_classes), 'num_classes must be specified'
         
         self._flat_features = MiniRocketFeatures(self.options.num_input_columns, self.options.num_input_steps, self.options.num_flat_features)
         get_minirocket_features()
         
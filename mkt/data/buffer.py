
import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from typing import *
from pprint import pprint

class TensorBuffer:
   _capacity:int
   _pos:int
   item_shape:tuple
   
   def __init__(self, size:int, item_shape:Optional[tuple]=None, dtype=None, ndim=None) -> None:
      self._capacity = size
      self.item_shape = item_shape
      self._pos = 0
      self._dtype = dtype
      self._ndim = (ndim if ndim is not None else (len(item_shape) if item_shape is not None else None))
      
      if item_shape is not None:
         self.data = np.empty((size, *item_shape), dtype=self._dtype)
      
      else:
         self.data = None
         
   def _prepush(self, el:ndarray):
      if self.data is None:
         if self.item_shape is None:
            if self._ndim is not None:
               ndim = self._ndim
               if el.ndim == ndim:
                  self.item_shape = tuple(el.shape)
               elif el.ndim == ndim+1:
                  self.item_shape = tuple(el.shape[1:])
               else:
                  raise ValueError(f'{ndim} != {el.ndim}')
            else:
               self.item_shape = tuple(el.shape)
               
         assert self.item_shape is not None   
         
         #* create the ndarray-buffer (`self.data`)
         self.data = np.empty((self._capacity, *self.item_shape), dtype=self._dtype)
      assert self.data is not None, 'no data buffer present to write data onto :C'
   
   def push(self, el:ndarray):
      self._prepush(el)
      
      self.grow()
      
      #* if the number of dimensions on `el` matches the number of dimensions specified in `self.item_shape`
      if el.ndim == len(self.item_shape):
         #* then we can simply push that fucker onto the buffer
         self.data[self._pos] = el
         self._pos += 1
      
      #* if `el` seems to be an array of elements which would satisfy the above condition
      elif el.ndim == len(self.item_shape)+1:
         #* then iterate over each value and push that value
         for i in range(el.shape[0]):
            self.push(el[i])
      
      else:
         #? complain about the mal-shaped input
         raise ValueError(f'Expected {self.item_shape}, got {el.shape}')
      
      #* return the new length of the buffer
      return self._pos
   
   def pop(self):
      self._pos -= 1
      r = self.data[self._pos]
      return r
   
   def shift(self):
      first = self.data[0]
      for i in range(1, self._pos):
         self.data[i-1] = self.data[i]
      return first
   
   def unshift(self, el:ndarray):
      for i in range(self._pos-1, 1, -1):
         self.data[i+1] = self.data[i]
      self.grow()
      self.data[0] = el
      self._pos += 1
      return self._pos
   
   def grow(self, mult:int=2):
      #? check whether the buffer is full (needs to grow)
      if self._pos == len(self.data):
         _d:ndarray = self.data
         new_cap = (self._capacity * mult)
         data_size = len(self.data)
         
         self.data = np.empty((new_cap, *self.item_shape))
         self._capacity = new_cap
         self.data[:data_size] = _d
      
      return self
   
   def get(self)->ndarray:
      return self.data[:self._pos]
   
   def tcpy(self)->Tensor:
      return torch.from_numpy(self.data.copy())
   
   @property
   def T(self)->Tensor:
      return torch.from_numpy(self.get())
   
   def is_full(self)->bool:
      return (self._pos >= self._capacity)
   
   def capacity(self)->int:
      return (self._capacity - self._pos - 1)
   
   def size(self)->int:
      return self._capacity
   
   @property
   def dtype(self):
      if self._dtype is None and self.data.dtype is not None:
         t = self._dtype = self.data.dtype
      else:
         t = self._dtype
      
      assert t is not None
      
      if t.name == 'object':
         pprint('object dtype strongly recommended against!')
      
      return t
   
   def __len__(self)->int:
      return self._pos
   
   @property
   def shape(self)->tuple:
      return (self._pos, *self.item_shape)

from fastai.callback.core import *
from mkt.tools import once

class BacktestingCallback(Callback):
   @once
   def after_epoch(self):
      # print([x for x in dir(self) if not x.startswith('__') and not x.endswith('__')])
      mdl = self.learn.model
      print(mdl)
      
   

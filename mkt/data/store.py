
from typing import *
from pandas import DataFrame
import pandas as pd
from cachetools import cached, Cache, keys
import sys, os
P = os.path
from pathlib import Path

#TODO: create StockLoader class with these methods + configurability

def list_stonks(stonk_dir='./stonks', shuffle=True)->List[str]:
   stonk_dir = './stonks'
   from pathlib import Path
   from random import shuffle
   
   tickers = [str(P.basename(n))[:-8] for n in Path(stonk_dir).rglob('*.feather')]
   shuffle(tickers)
   
   return tickers


@cached(cache=Cache(200))
def load_frame(sym:str, dir='./stonks')->DataFrame:
   dir = './stonks'
   #* read the DataFrame from the filesystem
   df:DataFrame = pd.read_feather(P.join(dir, '%s.feather' % sym))
   
   #* iterating forward through the frame, replace all NA values with the last non-NA value
   df = df.fillna(method='ffill')
   #* iterating backward through the frame, replace all NA values with the last non-NA value
   df = df.fillna(method='bfill')
   #* drop any remaining rows containing NA values
   df = df.dropna()
   
   #* reindex the frame by datetime
   df = df.set_index('datetime', drop=False)
   df.name = sym
   
   return df
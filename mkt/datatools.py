from collections import UserString
from time import sleep
from numba import jit, float32
from numba import njit as _njit

import numpy as np
from typing import *
import os
import sys
import math
import random
import re

P = os.path

from numpy import ndarray

base = P.dirname(os.getcwd())
print(base)
sys.path.append(base)

from tqdm import tqdm
import pandas as pd
from pandas import DataFrame

# from sklearn.preprocessing import MinMaxScaler, minmax_scale
from mkt.tools import unzip
from cytoolz import *
# from kraken import krakenClient
from pathlib import Path
import torch
from torch import Tensor

def renormalize(n, r_a, r_b):
   delta1 = r_a[1] - r_a[0]
   delta2 = r_b[1] - r_b[0]
   return (delta2 * (n - r_a[0]) / delta1) + r_b[0]

def quantize(n:Tensor, nQ:int, r_a=(0.0, 1.0), r_b=None, rounding_fn=torch.ceil):
   n = n.detach()
   if True in n[n == torch.nan]:
      raise Exception('no')
   n[n < r_a[0]] = r_a[0]
   n[n > r_a[1]] = r_a[1]
   scaled = renormalize(n, r_a, (0, nQ-1))
   return rounding_fn(scaled)
   
stablecoins = set([
   'USDT', 'USDC', 'UST',
   'WBTC', 'TBTC'
])

fiat_currencies = set(('JPY', 'AUD', 'CAD', 'EUR', 'GBP', 'CHF', 'AED', 'USD'))

from pprint import pprint

def build_cache(saveto=None):
   prog = tqdm(selected_symbols)
   df_cache = {}
   
   for sym in prog:
      pair_id = sym2id[str(sym)]
      if pair_id is None:
         print(f'No pair found for "{sym}"')
         continue
      ohlc, _ = client.get_ohlc_data(pair_id, interval=1440)
      df_cache[str(sym)] = ohlc
      
   if saveto is not None:
      import pickle
      with open(saveto, 'wb+') as f:
         pickle.dump(df_cache, f)
      
   return df_cache

def get_cache(p='./.kcache/all.pickle'):
   import pickle
   
   if not os.path.exists(p):
      cache = build_cache(saveto=p)
   else:
      cache = pickle.load(open(p, 'rb'))
      
   return cache

def get_cache_buffer(symbols:Iterable[str], columns:Iterable[str], cache:Optional[Dict[str, pd.DataFrame]]=None, select=None, n_samples=365, tolerant=False):
   if cache is None:
      cache = get_cache()
   symbols = set(symbols)
   for sym in symbols:
      if sym not in cache:
         # cache[sym] = client.get_ohlc_data(sym2id[str(sym)], interval=1440)
         if tolerant:
            symbols.remove(sym)
         else:
            raise ValueError(f'Symbol("{sym}") not present in the cache')
   
   loaded = []
   scalers = []
   buffer = []
   assert columns is not None and len(columns) != 0, 'must provide column names via columns argument'
   if isinstance(columns, str):
      columns = list(map(lambda s: s.strip(), columns.split(','))) if ',' in columns else columns.split(' ')
   
   for symbol in symbols:
      df = cache[symbol]
      if not all((c in df.columns) for c in columns):
         raise ValueError(f'DataFrame for "{symbol}" is missing column "{column}"')
      
      if len(df) < n_samples:
         if tolerant:
            # df = df.copy(deep=True)
            # ts = df.index.to_numpy()
            new_index = pd.date_range(df.index.max() - pd.DateOffset(days=n_samples), df.index.max())
            dummy_ohlc = np.full((n_samples - len(df), len(df.columns)), pd.NA)
            df = pd.DataFrame(data=np.vstack((dummy_ohlc, df.to_numpy())), index=new_index, columns=columns)
         else:
            raise ValueError(f'.. on "{symbol}", expected at least {n_samples} rows, got {len(df)}')
      else:
         df = df[columns].tail(n_samples)
      if select is not None:
         df = select(df)
      loaded.append(symbol)
      
      dfnp = df.to_numpy()
      
      scaler = MinMaxScaler()
      scalers.append(scaler)
      avg = np.mean(dfnp[:, :-1], axis=1).reshape(-1, 1)
      scaler.fit(avg)
      # dfnp = minmax_scale(dfnp)
      dfnp = renormalize(dfnp, (dfnp.min(), dfnp.max()), (0.0, 1.0))
      buffer.append(dfnp)
   
   np_buffer = np.asanyarray(buffer, dtype='float32')
     
   return loaded, scalers, np_buffer

def calc_Xy(cache: Dict[str, pd.DataFrame], columns=None, nT_in=365, nT_out=60):
   X = []
   y = []
   
   all_symbols = list(set(cache.keys()))
   symbols = []
   assert columns is not None
   nC = len(columns)
   
   buffers = []
   
   for k, sym in enumerate(all_symbols):
      df = cache[sym][columns]
      df = df.fillna(method='ffill').fillna(method='bfill').dropna()
      df_np:ndarray = df.to_numpy()
      
      if len(df_np) >= nT_in+nT_out:
         # if len(df_np) > nT_in:
            # df_np = df_np[:-nT_in]
         buffers.append(df_np)
         symbols.append(sym)
      else:
         continue
   
   # assert all(len(a)==nT_in for a in buffers)
   
   nK = len(symbols)
   ground = np.zeros((nK, nC))
   for k in range(nK):
      ground[k] = buffers[k][-nT_out-1, :]
   
   b_ranges = []
   move_ranges = []
   
   totalsize = len(buffers[0])
   for i in range(1, len(buffers)):
      totalsize = min(totalsize, len(buffers[i]))
   nB = len(list(range(1+nT_in, totalsize-nT_out)))
   print('batch_size=', )
   # for j, b in enumerate(buffers):
   #    buffers[j] = b[:len(buffers[0]*nB)]
   
   X = np.zeros((nB, nK, nT_in, nC))
   y = np.zeros((nB, nK, nT_out, nC))
   
   for b in range(1+nT_in, totalsize-nT_out):
      for k in range(len(symbols)):
         buffer = buffers[k]
         assert len(buffer) >= totalsize-1
         _min, _max = buffer.min(), buffer.max()
         b_ranges.append((_min, _max))
         
         buffer = buffers[k] = renormalize(buffer, (_min, _max), (0.0, 1.0))
         
         # assert len(buffer) == nT_in+nT_out
         
         _x:ndarray = buffer[b-nT_in-1:b-1]
         _y:ndarray = np.expand_dims(buffer[b-1], 0)

         print(_x.shape, _y.shape)
         
         assert _x.shape == X.shape[2:], f'{_x.shape} != {X.shape[2:]}'
         assert _y.shape == y.shape[2:], f'{_y.shape} != {y.shape[2:]}'
         
         X[b-1-nT_in, k] = _x
         y[b-1-nT_in, k] = _y
         
         print(f'x[k].shape={_x.shape}', f'y[k].shape={_y.shape}')
      
   return (
      symbols,
      b_ranges,
      buffers,
      move_ranges,
      ground,
      X,
      y
   )

import pickle
from cytoolz import *

@curry
def ohlc_resample(freq:str, df:DataFrame):
   G = None
   
   if freq == 'D':
      G = df.datetime.dt.date
   else:
      raise Exception(f'Invalid frequency {freq}')
   
   sampler = df.groupby(G)
   
   def ohlc(g: DataFrame):
      idx = g.datetime.iloc[0].date
      O = g.open.iloc[0]
      C = g.close.iloc[-1]
      L = g.low.min()
      H = g.high.max()
      V = g.volume.sum()
      
      return pd.Series(data=[O, H, L, C, V], index=['open', 'high', 'low', 'close', 'volume'], name=idx)
   
   rows = sampler.apply(ohlc)
   return rows

def unpack(packed_path: str):
   (symbols, indexes, frames) = pickle.load(open(packed_path, 'rb'))
   result = {}
   for sym in symbols:
      df:DataFrame = frames[sym]
      # print(df)
      if 'datetime' in df.columns:
         df = df.set_index('datetime', drop=True)
      result[sym] = df
      
   return result

def load_ts_dump(from_folder='./nasdaq_100'):
   man = pickle.load(open(P.join(from_folder, 'manifest.pickle'), 'rb'))
   syms = list(man.keys())
   result = {}
   
   for sym in syms:
      df:pd.DataFrame = pd.read_feather(P.join(from_folder, f'{sym}.feather'))
      df.set_index('datetime', inplace=True, drop=True)
      df = df.resample('D').mean()
      df = df.interpolate(method='linear').fillna(method='bfill').dropna()
      result[sym] = df
   
   return result

def mk_hrs2tmrw_ds():
   hourly = unpack('./sp100_hourly.pickle')
   daily:Dict[str, DataFrame] = unpack('./sp100_daily.pickle')

   symbols = list(hourly.keys())
   n_days = np.array([len(d) for d in daily.values()]).max()
   assert set(hourly.keys()) == set(daily.keys())

   # daily_hours = np.zeros((len(symbols), n_days, 7, 5), 'float32')
   # daily_tomorrow_summary = np.zeros((len(symbols), n_days, 4), dtype='float32')

   daily_hours, daily_tomorrow_summary = TensorBuffer(100000, (7, 5)), TensorBuffer(100000, (4,))

   daysidx:Optional[pd.DatetimeIndex] = None

   for d in daily.values():
      if len(d) == n_days:
         daysidx = d.index

   assert daysidx is not None

   dates:np.ndarray = daysidx.values
   skipped = []

   for i, symbol in enumerate(symbols):
      hours = hourly[symbol]
      hours = hours.pct_change()[1:]
      
      hours['date'] = hours.index
      hours['date'] = hours['date'].dt.date
      
      days = daily[symbol]
      days = days.pct_change()[1:]
      
      for j, date in enumerate(days.index):
         dhd = hours['date']
         day_hours:DataFrame = hours[hours.date == date][['open', 'high', 'low', 'close', 'volume']]
         day = days.loc[date]
         
         if len(day_hours) < 7:
            # print(f'Skipping {date} due to insufficient number of hours for the trading day')
            skipped.append((symbol, day_hours.index))
            continue
         
         elif (True in pd.isna(day_hours)) or (True in pd.isna(day)):
            # print(f'Skipping {date} due to null values')
            skipped.append((symbol, day_hours.index))
            continue
         
         elif j == len(days.index)-2:
            continue
         
         day_hours = day_hours.to_numpy()
         (date_where,) = np.where(dates == date)
         # daily_hours[i, date_where, :, :] = day_hours
         daily_hours.push(day_hours)
         
         if date_where[0] >= (len(dates) - 2):
            continue
         
         next_date  = np.datetime64(str(dates[date_where+2][0]))
         next_date2 = np.datetime64(str(days.loc[date:].index[0]))
         
         # assert next_date2 == next_date, f'{next_date2} != {next_date}'
         
         if next_date > dates.max() or next_date > days.index.max():
            continue
         
         tomorrow = days.loc[(days.index == next_date)|(days.index == next_date2)].iloc[0].to_numpy().squeeze()
         # daily_tomorrow_summary[i, date_where, :] = tomorrow[:-1]
         daily_tomorrow_summary.push(tomorrow[:-1])
         
   return daily_hours.T, daily_tomorrow_summary.T

def load_hrs2tmrw_ds():
   import os
   P = os.path
   ds_path = './today_hours2tomorrow_ohlc.pickle'
   if P.exists(ds_path):
      daily_hours, daily_tomorrow_summary = pickle.load(open(ds_path, 'rb'))
   else:
      daily_hours, daily_tomorrow_summary = mk_hrs2tmrw_ds()
      pickle.dump((daily_hours, daily_tomorrow_summary), open(ds_path, 'wb'))
      
   X, fy = daily_hours, daily_tomorrow_summary
   return X, fy

def percent_change(arr):
   result = np.diff(arr)/arr[1:]*100
   return result

def pl_binary_labeling(y: ndarray):
   labels = np.zeros((len(y), 2))
   
   ydelta = percent_change(y)
   thresh = 0.06 #* (0.006%)
   
   E = np.argwhere((ydelta <= thresh)&(ydelta >= -thresh))
   P = np.argwhere(ydelta > thresh)
   L = np.argwhere(ydelta < -thresh)
   
   labels[L, 0] = 1.0 # loss (-1)
   labels[P, 1] = 1.0 # profit (1)
   
   return labels

def pl_trinary_labeling(y:ndarray, thresh:float=0.006, fmt=1):
   if fmt == 2:
      ydelta = percent_change(y)
      P = np.argwhere(ydelta > thresh)
      L = np.argwhere(ydelta < -thresh)
      E = np.argwhere((ydelta <= thresh)&(ydelta >= -thresh))
      n_classes = 3
      buckets = [E, L, P]
      labels = np.zeros((len(y),))
      for i, bucket in enumerate(buckets):
         labels[bucket] = renormalize(i, (0, 2), (0.0, 1.0))
      return labels
   
   labels = np.zeros((len(y), 2))
   
   ydelta = percent_change(y)
   
   P = np.argwhere(ydelta > thresh)
   L = np.argwhere(ydelta < -thresh)
   E = np.argwhere((ydelta <= thresh)&(ydelta >= -thresh))
   
   labels[L, 0] = 1.0 # loss (-1)
   labels[P, 1] = 1.0 # profit (1)
   labels[E, 2] = 1.0 #
   
   return labels


from torch import Tensor
from torch.autograd import Variable

def rescale(x, new_min=0.0, new_max=1.0):
   old_range = (x.min(), x.max())
   return old_range, renormalize(x, old_range, (new_min, new_max))

def norm_batches(x: Tensor):
   with torch.no_grad():
      res = Variable(torch.zeros_like(x))
      for i in range(x.size(0)):
         batch = x[i]
         _, scaled_batch = rescale(batch, 0.0, 1.0)
         res[i] = scaled_batch
      return res
   
# def training_suitability(df: DataFrame)->float:
#    # from math import 
#    d = df['close'].pct_change().iloc[1:]
#    U = d[d > 0]
#    D = d[d < 0].abs()
   
#    mP, mL = U.mean(), D.mean()
   
#    U = []
#    D = []
   
#    for i in range(len(d)):
#       mv = d[i]
#       if mv > 0:
#          U.append(mv)
#       elif mv < 0:
#          D.append(abs(mv))
         
#    U, D = np.array(U), np.array(D)
#    nU, nD = len(U), len(D)
   
#    return (nU / nD)

def training_suitability(df: DataFrame):
   d = df['close'].pct_change().iloc[1:]
   thresh = 0.006
   
   U = d[d > thresh]
   D = d[d < -thresh].abs()
   
   mP, mL = U.mean(), D.mean()
   nP, nL, nA = len(U), len(D), len(df)-1
   coverage = (nP + nL)/nA
   return coverage
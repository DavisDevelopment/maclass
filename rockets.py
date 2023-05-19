from pprint import pprint
from pandas import Series
from termcolor import colored
from mkt.data import *
import gc


from mkt.tools import nor
from tsai.basics import *
import sktime
import sklearn
from faux.backtesting.common import samples_for, split_samples

# from tsai.models.MINIROCKET import *
from tsai.models.MINIROCKET import *

from cytoolz.dicttoolz import valmap, valfilter, keymap, keyfilter

from rocket_model import RocketShipOptions, RocketShip

def testonabunchofstuff():
   # Univariate classification with sklearn-type API
   for dsid in get_UCR_multivariate_list():
      fdsid = colored(dsid, 'cyan', attrs=['bold'])
      
      print(f'Loading {fdsid} dataset...')
      try:
         #* Download the dataset
         X_train, y_train, X_valid, y_valid = get_UCR_data(dsid)
      except FileNotFoundError:
         continue
      except KeyboardInterrupt:
         print(colored('Skipped ', 'red'), fdsid)
         continue
         
      if X_train is None:
         continue

      # Computes MiniRocket features using the original (non-PyTorch) MiniRocket code.
      # It then sends them to a sklearn's RidgeClassifier (linear classifier).
      model = MiniRocketClassifier(
         num_features = 10000,
         max_dilations_per_kernel = 32,
         random_state = None,
         alphas = np.logspace(-3, 3, 7),
         normalize_features = True,
         memory = None,
         verbose = False,
         scoring = None,
         class_weight = None
      )
      timer.start(False)
      
      print(f'Fitting model to {fdsid}...')
      model.fit(X_train, y_train)
      t = timer.stop()
      print(f'({fdsid}) valid accuracy    : {model.score(X_valid, y_valid):.3%} time: {t}')
      
      #* save the model to disk
      # print(f'Saving pretrained model for the {fdsid} dataset')
      # model.save(f'./_pretrained/{dsid}.pkl')

YCOL = '$LABEL'

def extract_features_from(options:RocketShipOptions, df:DataFrame):
   assert df.name is not None, 'DataFrame must have a name!'
   
   df:DataFrame = df.drop(columns=['datetime', 'volume'])
   
   pmv:Series = df['close'].pct_change()
   
   # Define the boundaries of each bucket
   boundaries = nor(options.label_boundaries, np.arange(-0.5, 0.75, 0.05))
   print(boundaries)
   print('No. of classes: ', len(boundaries))
   
   # Use pandas.cut to get the index of the bucket that each price move falls into
   bucket_indices = pd.cut(pmv, bins=boundaries, labels=False, right=False)
   
   # Shift the indices to represent buckets ranging from 0 to 200
   bucket_indices -= int(boundaries[0] / 0.05)

   # Clip the indices to ensure they're within the valid range
   bucket_indices = np.clip(bucket_indices, 0, 200)

   # Use the indices as the labels
   label = pd.Series(bucket_indices, index=df.index, name=YCOL)

   df[YCOL] = label
   df = df.dropna()
   df[YCOL] = df[YCOL].astype('int')
   
   #TODO: replace ('open', 'high', 'low', 'close', 'volume') columns with their respective `.pct_change()`s
   
   return df
   
def extract_samples_from(df: DataFrame, options:RocketShipOptions, date_begin=None, date_end=None, for_training=True, train_pct=0.85, window_size=10):
   assert df.name is not None, 'DataFrame must have a name!'
   from sklearn.preprocessing import QuantileTransformer
   
   ts_idx, X, y = samples_for(df, analyze=partial(extract_features_from, options), xcols=None, x_timesteps=window_size, y_type='pl_binary', y_lookahead=1, y=YCOL, scaler=QuantileTransformer, scaling=True, scale_y=False)
   # idx:pd.DatetimeIndex = pd.DatetimeIndex([d.date() for d in ts_idx])
   
   if for_training:
      # split the data
      (X,) = tuple((np.expand_dims(v, 1) if v.ndim == 2 else v.swapaxes(1, 2)) for v in (X,))
      train_idx, train_X, train_y, test_idx, test_X, test_y = split_samples(index=ts_idx, X=X, y=y, pct=train_pct, shuffle=False)
      
      return (ts_idx, train_X, train_y, test_X, test_y)
   
   else:   
      #split the data 
      if date_begin is not None or date_end is not None:
         begin_index = None
         end_index = None
         
         for i, ts in enumerate(ts_idx):
            if date_begin is not None and begin_index is None and ts >= date_begin:
               begin_index = i
            
            if date_end is not None and end_index is None and ts >= date_end:
               end_index = i
               break
               
         ts_idx = ts_idx[begin_index:end_index]
         X = X[begin_index:end_index]
         y = y[begin_index:end_index]
      
      #* train the model on 85% of the available data, evaluating on the remaining 15%
      # train_X, train_y, test_X, test_y = split_samples(X=X, y=y, pct=val_split, shuffle=False)
      # train_X, test_X = tuple((v.unsqueeze(1) if v.ndim == 2 else v.swapaxes(1, 2)) for v in (train_X, test_X))
      (X,) = tuple((np.expand_dims(v, 1) if v.ndim == 2 else v.swapaxes(1, 2)) for v in (X,))
      
      return (
         ts_idx,
         X,
         y
      )
      
from mkt.data.buffer import TensorBuffer

def dataset_from_dir(options:RocketShipOptions, dp='./stonks', saveto=None):
   import os
   P = os.path
   allstonks = list_stonks(dp)
   # print(len(allstonks), 'stonks')
   if saveto is None:
      saveto = f'{P.basename(dp)}_samples.pkl'
   
   if P.exists(saveto):
      try:
         result = (train_X, train_y, test_X, test_y) = pickle.load(open(saveto, 'rb'))
         
         return result
      except:
         pass
         
   Xb_train = TensorBuffer(10_000, ndim=2)
   yb_train = []
   
   Xb_test = TensorBuffer(10_000, ndim=2)
   yb_test = []

   for sym in allstonks:
      print(sym)
      #TODO: load features for this symbol, and append those features to a buffer
      df:DataFrame = load_frame(sym, dir=dp)
      idx, train_X, train_y, test_X, test_y = extract_samples_from(df, options)
      
      Xb_train.push(train_X)
      Xb_test.push(test_X)
      
      yb_train.append(train_y)
      yb_test.append(test_y)
      
   train_y = np.concatenate(yb_train)
   test_y = np.concatenate(yb_test)
   
   train_X = Xb_train.get()
   test_X = Xb_test.get()

   dump = (train_X, train_y, test_X, test_y)
   pickle.dump(dump, open(saveto, 'wb+'))
   return dump

from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *



def minirocket_features(X, splits, save_as=None, save_model_as=None):
   #* restore cached features when possible
   if save_as is not None and P.exists(save_as):
      X_feat = pickle.load(open(save_as, 'rb'))
      if len(X_feat) == len(X):
         return X_feat
   
   #* compute MiniRocket features
   mrf = MiniRocketFeatures(X.shape[1], X.shape[2])
   X = X.astype('float32')
   X_train = X[splits[0]]
   mrf.fit(X_train)
   X_feat = get_minirocket_features(X, mrf, chunksize=1024, to_np=True)
   print('MiniRocket features shape:', X_feat.shape)
   
   
   #* cache MiniRocket features for next time
   if save_as is not None:
      pickle.dump(X_feat, open(save_as, 'wb+'))
   
   #* cache MiniRocketFeatures for future use
   if save_model_as is not None:
      save_path = Path(save_model_as)
      print(save_path.suffix)
      save_path.parent.mkdir(parents=True, exist_ok=True)
      torch.save(mrf.state_dict(), save_path)
      
   return X_feat

def train_mr_head(train_X, train_y, test_X, test_y):
   # As above, use tsai to bring X_feat into fastai, and train.
   from tsai.data.validation import get_splits

   # Convert to X, y, splits
   X, y, splits = combine_split_data([train_X, test_X], [train_y, test_y])

   # Print the shapes of the resulting arrays
   print(X.shape)       # (n_samples, ...)
   print(y.shape)       # (n_samples,)
   print(len(splits))   # 2
   
   #TODO: also return the MiniRocketFeatures object from this function
   X_feat = minirocket_features(X, splits, save_as='./_pretrained/minirocket_features.pkl')

   tfms = [None, TSClassification()]
   batch_tfms = TSStandardize(by_sample=True)
   dls = get_ts_dls(X_feat, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
   gc.collect()
   
   model = build_ts_model(MiniRocketHead, dls=dls)
   
   from cbs import BacktestingCallback
   
   learn = Learner(dls, model, metrics=accuracy, cbs=BacktestingCallback())
   
   lrf = learn.lr_find()
   print(dir(lrf))
   print('LR optima is:', lrf.valley)
   lr = lrf.valley
   print('lr=', lr)
   
   timer.start()
   # learn.fit_one_cycle(100, lr)
   learn.fit(20, lr)
   t = timer.stop()
   
   return learn

def train_model_on(ds):
   train_X, train_y, test_X, test_y = ds
   # minirocket_features(train_X)
   
   mdl = train_mr_head(train_X, train_y, test_X, test_y)
   print(mdl)
   
   return mdl

def equilabellize(options:RocketShipOptions, X, y):
   """
   extract subsets of (X, y), such that every label-class has approximately equal representation in the returned (sub_X, sub_y) values
   
   params:
     X: numpy array of shape (n_batches, n_steps, n_features)
     y: numpy array of shape (n_batches,)
   """
   
   import matplotlib.pyplot as plt
   
   # if not isinstance(y_arrays, tuple):
   #    y_arrays = (y_arrays,)
   
   bins = {}
   classes = np.unique(y).astype(int)
   min_cls = classes.min()
   cls_interval = (options.label_boundaries[1] - options.label_boundaries[0])
   
   lbl_tranlation = {}
   for c in classes:
      lbl_tranlation[c] = (c - min_cls)

   for c in classes:
      # c = c.astype(int)
      new_c = lbl_tranlation[c]
      matches = np.argwhere(y == c)
      y[matches] = new_c
      bins[new_c] = (matches, options.label_boundaries[new_c])
   
   significant_bins = valfilter(lambda b: len(b[0]) > 1_000, bins)
   binremap = {}
   kl = list(significant_bins.keys())
   
   for k, (matches, bound_max) in significant_bins.items():
      new_k = kl.index(k)
      y[matches] = new_k
      binremap[new_k] = (matches, (bound_max - cls_interval, bound_max))
   
   bins = binremap
   def fmtpct(n):
      s = ""
      if n < 0:
         pass
      elif n > 0:
         s += '+'
      return (s + f'{n:.2f}%')
   
   for lbl, (_, (min_delta, max_delta)) in bins.items():
      print(f'Label #{lbl}   =>   [{fmtpct(min_delta)}, {fmtpct(max_delta)}]')
   
   # # plot `bins` as a bar chart
   # plt.bar(significant_bins.keys(), [len(significant_bins[b][0]) for b in significant_bins.keys()])
   # plt.xlabel('Label Class')
   # plt.ylabel('Count')
   # plt.title('Label Class Distribution')
   # plt.show()
   X_chunks = []
   y_chunks = []
   for lbl, (indices, _) in bins.items():
      X_chunks.append(X[indices])
      y_chunks.append(y[indices])
   
   return (np.concatenate(X_chunks), np.concatenate(y_chunks))

if __name__ == '__main__':
   options = RocketShipOptions()
   options.num_input_steps = 10
   dataset = (train_X, train_y, test_X, test_y) = dataset_from_dir(options, './stonks')
   options.datafit(train_X, train_y)
   
   train_X, train_y = equilabellize(options, train_X, (train_y, test_y))
   # exit()
   print('No. of training samples:', len(train_X))
   print('No. of evaluation samples:', len(test_X))
   
   # X, y, splits = combine_split_data([train_X, test_X], [train_y, test_y])
   # Set the number of samples to subsample
   n_training_samples = 50_000
   n_eval_samples = 50_000

   # Generate a set of random indices
   train_indices = np.random.choice(train_X.shape[0], size=n_training_samples, replace=False)

   # Select the corresponding samples from X and y
   X_sub = train_X[train_indices]
   y_sub = train_y[train_indices]
   del dataset
   del train_X
   del train_y
   train_X = X_sub
   train_y = y_sub
   
   test_indices = np.random.choice(test_X.shape[0], size=n_eval_samples, replace=False)
   X_sub = test_X[test_indices]
   y_sub = test_y[test_indices]
   del test_X
   del test_y
   test_X, test_y = X_sub, y_sub
   
   dataset = (train_X, train_y, test_X, test_y)
   
   #* explicitly invoke GC
   gc.collect()
   
   print(len(train_X), len(train_y), len(test_X), len(test_y))
   
   learn = train_model_on(dataset)
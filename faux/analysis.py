#!/usr/bin/env python
# coding: utf-8
import json, tsai, builtins
from torch import Tensor
from numpy import ndarray
from pandas import DataFrame
from cytoolz.dicttoolz import valmap, valfilter, itemfilter, itemmap, merge
from typing import *
from faux.features.ta.loadout import Indicators, bind_indicator
from functools import *
from cytoolz.functoolz import *
import cytoolz.functoolz as ft
import pandas as pd
import pandas_ta as ta

from mkt.olbi import configurePrinting


def transform_json(json_data, variables={}, functions={}, path=[]):
    if isinstance(json_data, dict):
        return {k: transform_json(v, variables, functions, path + [k]) for k, v in json_data.items()}
    elif isinstance(json_data, list):
        return [transform_json(v, variables, functions, path + [i]) for i, v in enumerate(json_data)]
    elif isinstance(json_data, str):
        if "${" in json_data and "}" in json_data:
            return transform_json(json_data.format(**valmap(str, variables)))
        elif json_data == "?":
            return transform_json(input("Enter value for {}:".format(".".join(map(str, path)))), variables, functions, path)
        elif json_data.startswith("`") and json_data.endswith("`"):
            expr = json_data[1:len(json_data)-1]
            #TODO evaluate by parsing the expression to allow for some magic
            # print(json_data[1:][:-1]
            # return eval(expr, merge(functions, variables))
            return json_data
        else:
            return json_data
    else:
        return json_data
     
def parse_feature_spec(spec: List[Any])->Callable[[DataFrame], DataFrame]:
   analysis_funcs = list(map(parse_feature_spec_item, spec))
   
   def wfunc(df: DataFrame)->DataFrame:
      return reduce(lambda df, fn: fn(df), analysis_funcs, df)
   
   return wfunc

def lookup_fn(fn_lookup, fn_id):
   try:
      fn = bind_indicator(fn_id)
      return fn
   except Exception as error:
      pass
   raise KeyError(fn_id)

def bind_item_fn(func, configuration={}):
   @wraps(func)
   def wfunc(df:DataFrame, *args, **kwargs):
      return func(df, *args, **kwargs)
   return wfunc

class TAFnLookup:
   def __init__(self) -> None:
      self.cache = dict()
      
   def mount_pinescript(self, pinescript_or_filename:str):
      """ expose the definitions in the given pinescript as functions """
      #TODO
      return None
      
   def __getitem__(self, fn_id:str):
      # raise Exception(fn_id)
      if not isinstance(fn_id, str):
         return None
      
      elif fn_id in self.cache:
         return self.cache[fn_id]
      elif isinstance(fn_id, str) and hasattr(ta, fn_id):
         ta_fn = getattr(ta, fn_id)
         if callable(ta_fn):
            self.cache[fn_id] = ta_fn
            return ta_fn
      elif isinstance(fn_id, str) and hasattr(pd.Series, fn_id):
         series_method = getattr(pd.Series, fn_id)
         if callable(series_method):
            self.cache[fn_id] = series_method
            return series_method
      return None

fn_lookup = TAFnLookup()
import re

_uid = [0]
def anoncolid():
   # nonlocal _uid
   name = f'Unnamed {_uid[0]}'
   _uid[0] += 1
   return name

def parse_feature_spec_item(item:Any)->Callable[[DataFrame], DataFrame]:
   type_name = type(item).__qualname__
   if isinstance(item, str):
      if (item[0], item[-1]) == ('`', '`'):
         expr = item[1:-1]
         
         #TODO when the expr-string represents a column-assignment, return an eval_function that invokes eval inline
         assign_pat = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=\s*')
         m = assign_pat.match(expr)
         if m is not None:
            #* Named assignments
            def eval_fn(df: DataFrame):
               print(f'Evaluating {expr} on DataFrame..')
               df = df.copy()
               df.eval(expr, inplace=True)#, resolvers=(fn_lookup,))
               return df
            return eval_fn
         
         def eval_fn(df: DataFrame)->DataFrame:
            print(f'Evaluating {expr} on DataFrame..')
            result = df.eval(expr)#, resolvers=(fn_lookup,))
            df[anoncolid()] = result
            return df
         
         return eval_fn
      
   elif isinstance(item, list):
      if len(item) == 1:
         raise NotImplemented()
      
   elif isinstance(item, dict):
      if len(item) == 1:
         item_ident, item_args = next(iter(item.items()))
         item_fn = lookup_fn(fn_lookup, item_ident)
         item_wfn = bind_item_fn(item_fn, item_args)
         return item_wfn
      
   elif callable(item):
      return item
   
   raise ValueError(f'Unhandled {item}')

from faux.backtesting.common import load_frame
from mkt.tools import dotget, dotgets

json_data = {
   'name': 'Example (Proto)Strategy #1',
   'description': 'This is a test',
   
   'data': {
      'features': [
         # '`vol_delta=volume.pct_change()`',
         # '`z_close_delta = close.pct_change().zscore()`',
         # '`trend_a = close.pct_change().sma(5).zscore()`',
         '`lohi = (low + high) / 2`',
         '`spread = (high - low)`',
         
         {'bbands': {'length':10, 'mamode':'vwma', 'std':1.25}},
         {'rsi': {'length': 2}},
         {'atr': {'length': 10, 'mamode': 'vwma'}},
         {'stoch': {'k': 7, 'd': 3, 'smooth_k': 3, 'mamode': 'ema'}},
         {'obv': {}},
         {'bop': {}},
         
         '`lohi_delta = lohi.pct_change()`',
         '`lohi_delta.zscore()`',
         # '`pissAndTitties = (high - low)`',
         '`z_spread = spread.zscore(3)`',
      ]
   }
}

#? graft technical-analysis methods onto pandas Series
for name in dir(ta):
   val = getattr(ta, name)
   if callable(val) and not hasattr(pd.Series, name):
      setattr(pd.Series, name, val)

paramInterpolationPattern = re.compile(r'^\$(?:([\w][\w\d_]*)|(\{[\w][\w\d_]*\}))')


def parse_analysis_item(item:Any):
   if isinstance(item, str):
      s = str(item)
      while True:
         m = paramInterpolationPattern.match(s)
         if m is None:
            break
         print(m.group(1), m.group(2))

      # bound_code = lambda 

#TODO: extend to allow expression of a multiplicity of parameters for the analysis, which can then be filled in with concrete values post-hoc

class Analysis:
   _items:List[Any]
   indicators:List[Callable[[DataFrame], DataFrame]]
   
   def __init__(self, *items):
      if len(items) == 0:
         self._items = []
         self.indicators = []
      else:
         self.__setstate__(items)
      
      self._bm = {}
     
   def assign(self, column_name:str, dval=None):
      return self.define(column_name=column_name, dval=dval)
   
   def define(self, column_name:str, dval=None):
      """ add named column/variable to the analysis """
      if isinstance(dval, str):
         def putdatondare(df: DataFrame)->DataFrame:
            pd.eval
            putval = df.eval(dval, global_dict=dict(pd=pd), local_dict=dict(pd=pd))
            df[column_name] = putval
            return df
         
         self._items.append(f'`{column_name} = {dval}`')
         self.indicators.append(putdatondare)
         
      elif callable(dval):
         def putdatondare(df: DataFrame)->DataFrame:
            putval = dval(df)
            df[column_name] = putval
            return df
         
         self._items.append(dval)
         self.indicators.append(putdatondare)
         
      elif dval is None:
         def addemptycolumn(df: DataFrame):
            df[column_name] = None
            
            return df
         self.indicators.append(addemptycolumn)
         
      else:
         raise TypeError(dval)
      
      return self
      
   def add(self, item:Any):
      if callable(item):
         def wrapped_item(df: DataFrame)->DataFrame:
            #TODO move `wrapped_item` into the global namespace so that references to it can be pickled
            ret = item(df)
            #TODO handle `ret` being a non-DataFrame value
            return ret
         
         self._items.append(item)
         self.indicators.append(wrapped_item)
      
      elif isinstance(item, str):
         if (item[0], item[-1]) == ('`', '`'):
            item = item[1:-1]
         item = f'`{item}`'
         
         self._items.append(item)
         self.indicators.append(parse_feature_spec_item(item))
         
      else:
         self._items.append(item)
         self.indicators.append(parse_feature_spec_item(item))
         
      return self
   
   def __getstate__(self):
      return self._items
   
   def __setstate__(self, items):
      self._items = []
      self.indicators = []
      
      for item in items:
         self.add(item)
      
   def _fn(self, df:DataFrame):
      # steps = [df]
      for fn in self.indicators:
         df = fn(df)
         # steps.append(df)
      
      return df
      
   def __call__(self, df:DataFrame, **kwargs)->DataFrame:
      #TODO: flesh out with the same functionality found in `apply` (loadout.py)
      input_df:DataFrame = df.copy()
      output_df:DataFrame = self._fn(input_df)
      
      return output_df
   
   def is_parametric(self):
      for i in self.indicators:
         if isinstance(i, Parametric):
            return True
      return False
   
   def parameters(self):
      visited = set()
      for i in self._items:
         if isinstance(i, Parametric):
            for p in i._parameters:
               if p.name not in visited:
                  visited.add(p.name)
                  yield p
   
   def with_parameters(self, **kwargs):
      pdefs = list(self.parameters())
      params = {p.name:None for p in pdefs}
      
      for k, v in kwargs.items():
         if k in params:
            params[k] = v
            
      next_items = []
      for i in self._items:
         if isinstance(i, Parametric):
            # i.with_parameters(**params)
            next_items.append(i.substitute(**params))
         else:
            next_items.append(i)
      
      return Analysis(*next_items)
   
   def combosOfN(self, n:int=3):
      from itertools import combinations
      
      rawCombos = combinations(self._items, n)
      # print(list(rawCombos))
      
      # combos = map(set, rawCombos)
      
      for term_group in rawCombos:
         print(term_group)
         ag = Analysis(*term_group)
         print(ag)
         yield ag
   
   def __getattr__(self, name:str):
      if name in self._bm:
         return self._bm[name]
      elif self.is_indicator(name):
         bound = partial(bound_indicator, self, name)
         self._bm[name] = bound
         return bound
      raise AttributeError(name)
   
   def is_indicator(self, name:str):
      #TODO extend to other conditions as well
      return (hasattr(ta, name) and callable(getattr(ta, name)))
   
   
   def compile(self, **kwargs):
      import ast
      
      compilable = True
      failure_point = None
      
      for item in self._items:
         if isinstance(item, str):
            pass
         else:
            compilable = False
            failure_point = item
            break
      
      if (failure_point is not None):
         raise TypeError(f'Compilation failed at {failure_point}')
      
      #* == "Compile" to Python source code == *#
      src_nodes = []
      parse = lambda s: ast.parse(s, '_analysis_.py', mode='eval')
      
      for item in self._items:
         configurePrinting(tracing=False)
         
         item_expr_tree = parse(src_code)
         
         print('AST for "'+item+f'":  {item_expr_tree}')
         print(ast.dump(item_expr_tree, annotate_fields=True, include_attributes=True))
         
         src_nodes.append(item_expr_tree)
      
      

import ast
from ast import NodeVisitor as PyASTNodeVisitor, Subscript
from ast import NodeTransformer
from ast import AST as Node

class AnalysisTreePreprocessor(NodeTransformer):
   _uid = 1
   _source:Optional[List[Node]]
   
   def __init__(self, owner:Analysis):
      super().__init__()
      #TODO: handle scope properly
      #TODO: carry this concept to the next level with a Pinescript interpreter/compiler module to facilitate portability
      
      self.owner = owner
      self._source = None
      self._imports = None
      self._declarations = None
      self.__all__ = None
      
      self._df_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
      self._main_name = f'_analysis_{_uid}'
      AnalysisTreePreprocessor._uid += 1
      
   def process(self, nodes:List[Node])->List[Node]:
      self._source = nodes[:]
      output_body = [self.visit(node) for node in nodes]
      # return nodes
      self._main_function = ast.FunctionDef(
         name=self._main_name,
         args=ast.arguments(
            args=[
               # ast.arg(arg='self', annotation=None), # the Analysis object itself
               ast.arg(arg='G', annotation=None), # the Dict that would be passed to `pd.eval``
               ast.arg(arg='df', annotation=None) # the DataFrame
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
         ),
         body=[*output_body],
         decorator_list=[],
         returns=None
      )
      
      return [self._main_function]
   
   def visit_Name(self, node:ast.Name):
      """ 
      visit an Identifier node
       - 
      """
      name = node.id
      if name in self._df_columns:
         if isinstance(node.ctx, ast.Load):
            # Replace identifier with df[$id]
            return ast.Subscript(
               value=ast.Name(id='df', ctx=ast.Load()),
               slice=ast.Index(value=ast.Constant(value=name)),
               ctx=node.ctx
            )
      
      return super().visit_Name(node)
   
   def visit_Assign(self, node:ast.Assign):
      # Assign
      # ast.Assign(targets=[ast.Name(id='x', ctx=ast.Store())], value=ast.Constant(value=1))
      lhv = node.targets
      rhv = node.value
      
      if len(lhv) == 1:
         lhv = lhv[0]
         if isinstance(lhv, ast.Name):
            name = lhv.id
            if name not in self._df_columns:
               self._df_columns.append(name)
            lhv = self.visit_Name(lhv)
         elif isinstance(lhv, Subscript):
            #TODO selective column update
            pass
         else:
            raise ValueError(lhv)

         rhv = self.visit(rhv)
         
         return ast.Assign(targets=[lhv], value=rhv)

   
def bound_indicator(self, name:str, **options):
   o = {name: options}
   
   if self is None:
      return o
   else:
      self.add(o)

class Special:
   pass

class AnalysisParameter(Special):
   def __init__(self, name:str):
      # super().__init__()
      self.name = name
      #TODO: also hold information about the type and range of this parameter
      
class Parametric:
   _parameters:List[AnalysisParameter]
   _item:Any
   
   def __init__(self, item:Any, *parameters):
      self._parameters = list(parameters)
      self._item = item
      
   
   def substitute(self, item=None, **params):
      pnames = [p.name for p in self._parameters]
      subs = {}
      item = (self._item if item is None else item)
      
      for k, v in params.items():
         if k not in pnames:
            raise NameError(f'Expected on of {tuple(pnames)}, but got "{k}"')
         subs[k] = v
      
      return substitute(item, **subs)
   
from faux.pgrid import recursiveMap

def substitute(item=None, params={}):
   if isinstance(item, str):
      # self._item = self._item.format(**subs)
      s = str(item)
      while True:
         m = paramInterpolationPattern.match(s)
         if m is None:
            break
         print(m.group(1), m.group(2))
         ident = (m.group(1) + m.group(2))
         
         s = s.replace(m.group(0), params[ident])
         
      return s
   
   elif callable(item):
      return partial(item, **params)
   
   else:
      #TODO assert that `item` is a value of acceptible type
      return recursiveMap(item, partial(substitute, params=params))
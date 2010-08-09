import scipy.sparse as s
import numpy
from copy import deepcopy

from hydrat.preprocessor.features import FeatureMap
from hydrat.common.transform.weight import Discretize, TFIDF
def union(*fms):
  """Return a FeatureMap that is the union of two FeatureMaps"""
  if len(fms) == 1: return fms[0]
  #TODO: Sanity check
  fm = s.hstack([ f.raw for f in fms ])
  #TODO: Think about how to properly reconcile metadata
  metadata = dict()
  feature_desc = list()
  for f in fms:
    metadata.update(deepcopy(f.metadata))
    feature_desc.append(deepcopy(f.metadata['feature_desc']))
  if 'feature_uuid' in metadata: del metadata['feature_uuid']
  if 'feature_name' in metadata: del metadata['feature_name']
  metadata['feature_desc'] = tuple(feature_desc)
    
  return FeatureMap(fm.tocsr(), metadata) 

class UnsupervisedTransform(object):
  def __init__(self, transformer):
    self.transformer = transformer

  def __call__(self, fm):
    r = self.transformer.apply(fm)
    metadata = deepcopy(fm.metadata)
    name = self.transformer.__name__
    metadata['feature_desc'] += (name,)
    return FeatureMap(r, metadata)

def discretize(fm, coefficient=1000):
  t = UnsupervisedTransform(Discretize(coefficient=coefficient))
  return t(fm)

def tfidf(fm):
  t = UnsupervisedTransform(TFIDF())
  return t(fm)

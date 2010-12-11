import scipy.sparse as s
import numpy
from copy import deepcopy

from hydrat.preprocessor.features import FeatureMap
def union(*fms):
  """Return a FeatureMap that is the union of multiple FeatureMaps"""
  if len(fms) == 1: return fms[0]
  #TODO: Sanity check
  fm = s.hstack([ f.raw for f in fms ])
  #TODO: Think about how to properly reconcile metadata
  metadata = dict()
  feature_desc = tuple()
  for f in fms:
    metadata.update(deepcopy(f.metadata))
    feature_desc += deepcopy(f.metadata['feature_desc'])
  if 'feature_uuid' in metadata: raise ValueError, "Should not have feature_uuid"
  if 'feature_name' in metadata: raise ValueError, "Should not have feature_name"
  metadata['feature_desc'] = feature_desc
    
  return FeatureMap(fm.tocsr(), metadata) 


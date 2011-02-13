# datamodel.py
# Marco Lui February 2011
#
# This module contains hydrat's datamodel, which specifies the objects used by hydrat
# to represent different data.

import numpy
import scipy.sparse

class Fold(object):
  """
  Represents a fold. Abstracts accessing of subsections of a numpy/scipy array
  according to training and test indices.
  """
  def __init__(self, fm, train_ids, test_ids):
    self.fm = fm
    self.train_ids = train_ids
    self.test_ids = test_ids

  def __repr__(self):
    return '<Fold of %s (%d train, %d test)>' %\
      (str(self.fm), len(self.train_ids), len(self.test_ids))

  @property
  def train(self):
    return self.fm[self.train_ids]
    
  @property
  def test(self):
    return self.fm[self.test_ids]

class SplitArray(object):
  """
  Maintains a sequence of folds according to a given split (if any).
  """
  def __init__(self, raw, split=None, metadata = {}):
    self.raw = raw
    self.split = split
    self.metadata = deepcopy(metadata)
    if 'feature_desc' not in metadata:
      self.metadata['feature_desc'] = tuple()

  def __getitem__(self, key):
    return self.raw[key]

  def __repr__(self):
    return '<%s %s>' % (self.__class__.__name__, str(self.raw.shape))

  @property
  def split(self):
    return self._split

  @split.setter
  def split(self, value):
    self._split = value
    self.folds = []
    if value is not None:
      for i in range(value.shape[1]):
        train_ids = numpy.flatnonzero(value[:,i,0])
        test_ids = numpy.flatnonzero(value[:,i,1])
        self.folds.append(Fold(self, train_ids, test_ids))

class FeatureMap(SplitArray):
  """
  Represents a FeatureMap. The underlying raw array is a scipy.sparse.csr_matrix by convention
  """
  @staticmethod
  def union(*fms):
    # TODO: Sanity check on metadata
    if len(fms) == 1: return fms[0]

    fm = scipy.sparse.hstack([f[:] for f in fms])

    metadata = dict()
    feature_desc = tuple()
    for f in fms:
      metadata.update(deepcopy(f.metadata))
      feature_desc += deepcopy(f.metadata['feature_desc'])
    metadata['feature_desc'] = feature_desc

    return FeatureMap(fm.tocsr(), split=fms[0].split, metadata=metadata)
    

class ClassMap(SplitArray): 
  """
  Represents a ClassMap. The underling raw array is a numpy.ndarray with bool dtype by convention
  """
  pass

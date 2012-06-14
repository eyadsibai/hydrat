# datamodel.py
# Marco Lui February 2011
#
# This module contains hydrat's datamodel, which specifies the objects used by hydrat
# to represent different data.

import numpy
import scipy.sparse

from copy import deepcopy

import abc
from collections import Sequence

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
    # TODO: Not all splitarrays need a feature_desc!
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

  @staticmethod
  def stack(*fms):
    """
    stacking of instances
    """
    if len(fms) == 1: return fms[0]

    # The problem is that this can MemoryError out - it tries to allocate
    # enough memory to build the stacked version.
    fm = scipy.sparse.vstack([f[:] for f in fms])
    metadata = dict()
    feature_desc = tuple()

    metadata.update(deepcopy(fms[0].metadata))
    metadata['instance_space'] = tuple(f.metadata['instance_space'] for f in fms) 

    return FeatureMap(fm.tocsr(), metadata=metadata)
    

class ClassMap(SplitArray): 
  """
  Represents a ClassMap. The underling raw array is a numpy.ndarray with bool dtype by convention
  """
  @staticmethod
  def stack(*cms):
    """
    stacking of instances
    """
    if len(cms) == 1: return cms[0]

    cm = numpy.vstack([c[:] for c in cms])
    metadata = dict()
    feature_desc = tuple()

    metadata.update(deepcopy(cms[0].metadata))
    metadata['instance_space'] = tuple(c.metadata['instance_space'] for c in cms) 

    return ClassMap(cm, metadata=metadata)

###
# Task
###
class Task(object):
  __slots__ = [ 'train_vectors'
              , 'train_classes'
              , 'train_sequence'
              , 'test_vectors'
              , 'test_classes'
              , 'test_sequence'
              , 'metadata'
              , 'train_indices'
              , 'test_indices'
              , 'weights'
              ]

  def compute_weight(self, weight_function):
    if weight_function.__name__ not in self.weights:
      self.weights[weight_function.__name__] = weight_function(self.train_vectors, self.train_classes)
    return self.weights[weight_function.__name__]

class DataTask(Task):
  __slots__ = Task.__slots__ + [ 'class_map', 'feature_map', 'sequence']
  def __init__( self
              , feature_map
              , class_map
              , train_indices
              , test_indices
              , metadata
              , sequence = None
              ):
    if not issubclass(train_indices.dtype.type, numpy.int):
      raise ValueError, 'Expected integral indices'
    if not issubclass(test_indices.dtype.type, numpy.int):
      raise ValueError, 'Expected integral indices'

    self.class_map = class_map
    self.feature_map = feature_map
    self.train_indices = train_indices
    self.test_indices = test_indices
    # TODO: Sanity check on the partitioning of the sequence. There shouldn't be sequences
    #       that span train & test
    self.sequence = sequence
    self.metadata = dict(metadata)
    self.weights = {}

    
  @property
  def train_vectors(self):
    """
    Get training instances
    @return: axis 0 is instances, axis 1 is features
    @rtype: 2-d array
    """
    return self.feature_map[self.train_indices]

  @property
  def test_vectors(self):
    """
    Get test instances
    @return: axis 0 is instances, axis 1 is features
    @rtype: 2-d array
    """
    return self.feature_map[self.test_indices]

  @property
  def train_classes(self):
    """
    Get train classes 
    @return: axis 0 is instances, axis 1 is classes 
    @rtype: 2-d array
    """
    return self.class_map[self.train_indices]

  @property
  def test_classes(self):
    """
    Get test classes 
    @return: axis 0 is instances, axis 1 is classes 
    @rtype: 2-d array
    """
    return self.class_map[self.test_indices]

  @property
  def train_sequence(self):
    if self.sequence is None:
      return None
    else:
      indices = self.train_indices
      matrix = self.sequence[indices].transpose()[indices].transpose()
      return matrix

  @property
  def test_sequence(self):
    if self.sequence is None:
      return None
    else:
      indices = self.test_indices
      matrix = self.sequence[indices].transpose()[indices].transpose()
      return matrix

###
# TaskSet
###
class TaskSet(Sequence):
  """
  This represents the TaskSet interface. A taskset is basically a 
  sequence of tasks, and hence implements the Sequence ABC. It also
  carries an additional attribute "metadata". 
  """
  # TODO: Potentially introduce a metadata ABC
  @abc.abstractproperty
  def metadata(self):
    pass

  def __contains__(self, key):
    # this would require task-level equality. not sure where we could ever need this
    raise NotImplementedError("not clear when we need this or how it should behave")


class BasicTaskSet(TaskSet):
  def __init__( self, tasks, metadata):
    self.tasks = tasks
    self.metadata = dict(metadata)

  def __getitem__(self, key):
    return self.tasks[key]

  def __len__(self):
    return len(self.tasks)

  def __iter__(self):
    return iter(self.tasks)

  def __contains__(self, key):
    return key in self.tasks

# TODO: New-style Tasks
# TODO: What is a new-style Task???
class DataTaskSet(TaskSet):
  def __init__(self, featuremap, classmap, sequence=None, metadata={}):
    self.featuremap = featuremap
    self.classmap = classmap
    self.sequence = sequence
    self.metadata = dict(metadata)

  @classmethod
  def from_proxy(cls, proxy):
    """ Convenience method to build a DataTaskSet from a DataProxy """
    if proxy.split_name is None:
      raise ValueError, "cannot create taskset from proxy without a defined split"
    fm = proxy.featuremap
    cm = proxy.classmap
    sq = proxy.sequence
    md = proxy.desc
    return cls(fm, cm, sq, md)

  def __getitem__(self, key):
    fm = self.featuremap
    cm = self.classmap
    sq = self.sequence
    fold = fm.folds[key]
    DataTask(fm.raw, cm.raw, fold.train_ids, fold.test_ids, {'index':key}, sequence=sq)

  def __len__(self):
    return len(self.featuremap.folds)

###
# Result
###
from result.result import Result

###
# TaskSetResult
###
from result.tasksetresult import TaskSetResult

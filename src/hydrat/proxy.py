# proxy.py
# Marco Lui February 2011
#
# This class is meant to act as a go-between between the user and the dataset/store
# classes. It is initialized on a dataset (and an optional store), and provides
# convenience methods for accesing portions of that store that are directly
# impacted by the dataset API.

# TODO: Integrate the new SplitArray based FeatureMap and ClassMap back into store,
# and wherever else the old FeatureMap and ClassMap were used.
# TODO: strdoc
# TODO: __str__/__repr__

import logging
import os
import numpy

from copy import deepcopy

from hydrat.common import as_set
from hydrat.store import Store, StoreError, NoData, AlreadyHaveData
from hydrat.inducer import DatasetInducer
from hydrat.datamodel import FeatureMap, ClassMap

class DataProxy(object):
  def __init__( self, dataset, store=None,
        feature_spaces=None, class_space=None, split_name=None, sequence_name=None ):
    self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
    self.dataset = dataset

    if isinstance(store, Store):
      self.store = store
      work_path = os.path.dirname(store.path)
    elif store is None:
      # Open a store named after the top-level calling file
      import inspect
      stack = inspect.stack()
      filename = os.path.basename(stack[-1][1])
      store_path = os.path.splitext(filename)[0]+'.h5'
      self.store = Store(store_path, 'a')
    else:
      self.store = Store(store, 'a')

    self.inducer = DatasetInducer(self.store)

    self.feature_spaces = feature_spaces
    self.class_space = class_space
    self.split_name = split_name
    self.sequence_name = sequence_name

    # Hack to deal with bad interaction between pytables and h5py
    import atexit; atexit.register(lambda: self.store.__del__())

  @property
  def desc(self):
    md = dict()
    md['dataset'] = self.dsname
    md['instance_space'] = self.instance_space
    md['split'] = self.split_name
    md['sequence'] = self.sequence_name
    md['feature_desc'] = self.feature_desc
    md['class_space'] = self.class_space
    return md

    
  @property
  def dsname(self):
    return self.dataset.__name__

  @property
  def feature_spaces(self):
    """
    String or sequence of strings representing the feature spaces 
    to operate over.
    """
    return self._feature_spaces

  @feature_spaces.setter
  def feature_spaces(self, value): 
    value = as_set(value)
    if any( not isinstance(s,str) for s in value):
      raise TypeError, "Invalid space identifier: %s" % str(s)
    present_spaces = set(self.dataset.featuremap_names)
    if len(value - present_spaces) > 0:
      # Check if the missing spaces are already in the store
      missing = value - present_spaces
      unknown = []
      for space in missing:
        if not self.store.has_Data(self.dsname, space):
          unknown.append(space)
      if len(unknown) > 0:
        raise ValueError, "Unknown spaces: %s" % str(unknown)
    self._feature_spaces = value

  @property
  def featurelabels(self):
    """
    List of labels of the feature space
    """
    self.inducer.process(self.dataset, fms=self.feature_spaces)
    labels = []
    # TODO: Handle unlabelled (EG transformed) feature spaces
    for feature_space in sorted(self.feature_spaces):
      labels.extend(self.store.get_Space(feature_space))
    return labels

  @property
  def feature_desc(self):
    #TODO: Construct the complete feature_desc based on feature_spaces and any transforms applied.
    return self.feature_spaces

  @property
  def class_space(self):
    """
    String representing the class space to operate over.
    """
    return self._class_space

  @class_space.setter
  def class_space(self, value):
    if value is None:
      self._class_space = None
    else:
      if not isinstance(value, str):
        raise TypeError, "Invalid space identifier: %s" % str(value)
      present_classes = set(self.dataset.classmap_names)
      if value not in present_classes:
        raise ValueError, "Unknown space: %s" % value

      self._class_space = value

  @property
  def classlabels(self):
    self.inducer.process(self.dataset, cms=self.class_space)
    return self.store.get_Space(self.class_space)

  @property
  def split_name(self):
    return self._split_name

  @split_name.setter
  def split_name(self, value):
    if value is None:
      self._split_name=None
    else:
      if not isinstance(value, str):
        raise TypeError, "Invalid split identifier: %s" % str(value)
      present_splits=set(self.dataset.split_names)
      if value not in present_splits:
        raise ValueError, "Unknown split: %s" % value

      self._split_name = value

  @property
  def split(self):
    if self.split_name is None:
      return None
    self.inducer.process(self.dataset, sps=self.split_name)
    return self.store.get_Split(self.dsname, self.split_name)

  @property
  def sequence_name(self):
    return self._sequence_name

  @sequence_name.setter
  def sequence_name(self,value):
    if value is None:
      self._sequence_name=None
    else:
      if not isinstance(value, str):
        raise TypeError, "Invalid sequence identifier: %s" % str(value)
      present_sequences=set(self.dataset.sequence_names)
      if value not in present_sequences:
        raise ValueError, "Unknown sequence: %s" % value

      self._sequence_name = value

  @property
  def sequence(self):
    # TODO: Does this need to be in a SplitArray?
    if self.sequence_name is None:
      return None
    self.inducer.process(self.dataset, sqs=self.sequence_name)
    return self.store.get_Sequence(self.dsname, self.sequence_name)

  @property
  def instance_space(self):
    # Note that this cannot be set as it is implicit in the dataset
    return self.dataset.instance_space

  @property
  def instancelabels(self):
    return self.store.get_Space(self.instance_space)

  @property 
  def classmap(self):
    self.inducer.process(self.dataset, cms=self.class_space)
    cm = self.store.get_ClassMap(self.dsname, self.class_space)
    return ClassMap(cm.raw, split=self.split, metadata=cm.metadata)
   
  @property
  def featuremap(self):
    self.inducer.process(self.dataset, fms=self.feature_spaces)

    # TODO: Avoid this duplicate memory consumption
    featuremaps = []
    for feature_space in sorted(self.feature_spaces):
      # TODO: Get rid of this once we introduce new-style featuremaps
      #       into the store
      fm = self.store.get_FeatureMap(self.dsname, feature_space)
      featuremaps.append(FeatureMap(fm.raw, metadata=fm.metadata))

    # Join the featuremaps into a single featuremap
    fm = FeatureMap.union(*featuremaps)
    fm.split = self.split
    return fm

  def tokenize(self, tsname, extractor):
    """
    Map a feature extractor onto a tokenstream and save the corresponding
    output into the backing store.
    """
    # Definition of space name.
    space_name = '_'.join((tsname,extractor.__name__))
    if not self.store.has_Data(self.dsname, space_name):
      self.inducer.process(self.dataset, tss=tsname)

      # Read the tokenstream
      tss = self.store.get_TokenStreams(self.dsname, tsname)
      feat_dict = dict()

      # TODO: Backoff behaviour if multiprocessing fails
      #for i, id in enumerate(self.instancelabels):
      #  feat_dict[id] = extractor(tss[i])
      import multiprocessing as mp
      pool = mp.Pool(mp.cpu_count())
      tokens = pool.map(extractor, tss)
      for i, id in enumerate(self.instancelabels):
        feat_dict[id] = tokens[i]

      self.inducer.add_Featuremap(self.dsname, space_name, feat_dict)

    self.feature_spaces = space_name

  @property
  def taskset(self):
    self.store.new_TaskSet(FromProxy(self))
    return self.store.get_TaskSet(self.desc, None)

class TaskSet(object):
  """
  This base class represents the TaskSet interface. It consists of two
  attributes, desc and tasklist, which can be implemented as properties
  if lazy behaviour is desired.
  """
  desc = {}
  tasklist = []
  
# TODO: New-style Tasks
from hydrat.task import InMemoryTask
class DataProxyTaskSet(TaskSet):
  def __init__(self, proxy):
    self.proxy = proxy
    self.desc = self.proxy.desc

  @property
  def tasklist(self):
    fm = self.proxy.featuremap
    cm = self.proxy.classmap
    sq = self.proxy.sequence

    tasklist = []
    for i,fold in enumerate(fm.folds):
      tasklist.append(InMemoryTask(fm.raw, cm.raw, fold.train_ids, fold.test_ids, {'index':i}, sequence=sq))
    return tasklist

import hydrat.task.transform as tx
class Transform(TaskSet):
  def __init__(self, tasksetsource, transformer):
    self.tasksetsource = tasksetsource
    self.transformer = transformer

  @property
  def _desc(self):
    return tx.update_medata(tasksetsource.desc, self.transformer)
    
  @property
  def tasklist(self):
    # TODO: Work out how to get the required weights, and how to extend them back
    # TODO: Work out why we need add_args
    raise NotImplementedError
    self.weights = transformer.weights.keys()
    taskset = self.taskset
    new_taskset = tx.transform_taskset(taskset, transformer, add_args=add_args)
    self.store.extend_Weights(taskset)
    #self.store.new_TaskSet(new_taskset)


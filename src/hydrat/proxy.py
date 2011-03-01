# proxy.py
# Marco Lui February 2011
#

# TODO: Integrate the new SplitArray based FeatureMap and ClassMap back into store,
# and wherever else the old FeatureMap and ClassMap were used.
# TODO: strdoc
# TODO: __str__/__repr__

import logging
import os
import numpy
import scipy.sparse

from copy import deepcopy

from hydrat.common import as_set
from hydrat.store import Store, StoreError, NoData, AlreadyHaveData
from hydrat.inducer import DatasetInducer
from hydrat.datamodel import FeatureMap, ClassMap, TaskSet, DataTask

class DataProxy(TaskSet):
  """
  This class is meant to act as a go-between between the user and the dataset/store
  classes. It is initialized on a dataset (and an optional store), and provides
  convenience methods for accesing portions of that store that are directly
  impacted by the dataset API.
  """
  def __init__( self, dataset, store=None, inducer = None,
        feature_spaces=None, class_space=None, split_name=None, sequence_name=None,
        tokenstream_name=None):
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

    self.inducer = DatasetInducer(self.store) if inducer is None else inducer

    self.feature_spaces = feature_spaces
    self.class_space = class_space
    self.split_name = split_name
    self.sequence_name = sequence_name
    self.tokenstream_name = tokenstream_name

    # Hack to deal with bad interaction between pytables and h5py
    import atexit; atexit.register(lambda: self.store.__del__())

  @property
  def desc(self):
    # TODO: Eliminate this altogether
    return self.metadata

  @property
  def metadata(self):
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
    return tuple(sorted(self.feature_spaces))

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
  def tokenstream_name(self):
    return self._tokenstream_name

  @tokenstream_name.setter
  def tokenstream_name(self, value):
    if value is None:
      self._tokenstream_name=None
    else:
      if not isinstance(value, str):
        raise TypeError, "Invalid tokenstream identifier: %s" % str(value)
      present_tokenstream=set(self.dataset.tokenstream_names)
      if value not in present_tokenstream:
        raise ValueError, "Unknown tokenstream: %s" % value

      self._tokenstream_name = value

  @property
  def tokenstream(self):
    self.inducer.process(self.dataset, tss=self.tokenstream_name)
    return self.store.get_TokenStreams(self.dsname, self.tokenstream_name)

  @property
  def instance_space(self):
    # Note that this cannot be set as it is implicit in the dataset
    return self.dataset.instance_space

  @property
  def instancelabels(self):
    self.inducer.process(self.dataset)
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

  def tokenize(self, extractor):
    """
    Map a feature extractor onto a tokenstream and save the corresponding
    output into the backing store.
    """
    # Definition of space name.
    space_name = '_'.join((self.tokenstream_name,extractor.__name__))
    if not self.store.has_Data(self.dsname, space_name):
      # Read the tokenstream
      tss = self.tokenstream
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
  def tasks(self):
    fm = self.featuremap
    cm = self.classmap
    sq = self.sequence

    tasklist = []
    for i,fold in enumerate(fm.folds):
      t = DataTask(fm.raw, cm.raw, fold.train_ids, fold.test_ids, 
          {'index':i}, sequence=sq)
      tasklist.append(t)
    return tasklist

  @property
  def taskset(self):
    return self.store.new_TaskSet(self)

class CrossDomainDataProxy(DataProxy):
  metadata = {}
  def __init__(self, train_ds, eval_ds, store=None, feature_spaces=None,
        class_space=None, sequence_name=None, tokenstream_name=None):
    """
    Initialize on two datasets, one for training and one
    for evaluation. Note that we want these two to share
    the same Store. This behaves just like a DataProxy,
    but it bridges two datasets. The feature and class
    spaces stay the same, but the instance space is defined
    as the concatenation of the two.
    """
    self.train = DataProxy(train_ds, store, feature_spaces=feature_spaces,
        class_space=class_space, sequence_name=sequence_name, 
        tokenstream_name=tokenstream_name)
    self.eval = DataProxy(eval_ds, self.train.store, feature_spaces=feature_spaces,
        class_space=class_space, sequence_name=sequence_name, 
        tokenstream_name=tokenstream_name)

    self.feature_spaces = feature_spaces
    self.class_space = class_space
    self.sequence_name = sequence_name
    self.tokenstream_name = tokenstream_name


    self.inducer = self.train.inducer
    self.store = self.train.store

  @property
  def dsname(self):
    return '+'.join((self.train.dsname, self.eval.dsname))

  @property
  def feature_spaces(self):
    return self._feature_spaces

  @feature_spaces.setter
  def feature_spaces(self, value):
    self.train.feature_spaces = value
    self.eval.feature_spaces = value
    self._feature_spaces = as_set(value)

  @property
  def featurelabels(self):
    self.inducer.process(self.train.dataset, fms=self.feature_spaces)
    self.inducer.process(self.eval.dataset, fms=self.feature_spaces)
    labels = []
    for feature_space in sorted(self.feature_spaces):
      labels.extend(self.store.get_Space(feature_space))
    return labels

  @property
  def class_space(self):
    return self._class_space

  @property
  def classlabels(self):
    self.inducer.process(self.train.dataset, cms=self.class_space)
    self.inducer.process(self.eval.dataset, cms=self.class_space)
    return self.store.get_Space(self.class_space)

  @class_space.setter
  def class_space(self, value):
    self.train.class_space = value
    self.eval.class_space = value
    self._class_space = value

  @property
  def split_name(self):
    return 'crossdomain'

  @property
  def split(self):
    num_train = len(self.train.instancelabels)
    num_eval = len(self.eval.instancelabels)
    num_inst = num_train + num_eval
    retval = numpy.zeros((num_inst,1,2), dtype=bool)
    retval[:num_train,:,0] = True
    retval[-num_eval:,:,1] = True
    return retval

  @property
  def tokenstream_name(self):
    return self._tokenstream_name

  @tokenstream_name.setter
  def tokenstream_name(self, value):
    if value is None:
      self._tokenstream_name=None
    else:
      if not isinstance(value, str):
        raise TypeError, "Invalid tokenstream identifier: %s" % str(value)
      present_tokenstream_train=set(self.train.dataset.tokenstream_names)
      present_tokenstream_eval=set(self.eval.dataset.tokenstream_names)
      if value not in present_tokenstream_train:
        raise ValueError, "Unknown tokenstream: %s" % value
      if value not in present_tokenstream_eval:
        raise ValueError, "Unknown tokenstream: %s" % value

      self._tokenstream_name = value

  @property
  def tokenstream(self):
    return self.train.tokenstream + self.eval.tokenstream

  @property
  def instance_space(self):
    """ Returns a concatenation of the two instance spaces """
    return '+'.join((self.train.instance_space, self.eval.instance_space))

  @property
  def instancelabels(self):
    return self.train.instancelabels + self.eval.instancelabels

  @property
  def classmap(self):
    cm_train = self.train.classmap
    cm_eval = self.eval.classmap
    raw = numpy.vstack((cm_train.raw, cm_eval.raw))
    md = dict(dataset=self.dsname, class_space=self.class_space, 
        instance_space=self.instance_space)
    return ClassMap(raw, split=self.split, metadata=md)

  @property
  def featuremap(self):
    # NOTE: We access the featurelabels of both in order to ensure that
    # full common feature space is learned before we attempt to access
    # the actual featuremaps
    self.train.featurelabels
    self.eval.featurelabels

    fm_train=self.train.featuremap
    fm_eval=self.eval.featuremap
    raw = scipy.sparse.vstack((fm_train.raw, fm_eval.raw)).tocsr()
    md = dict(dataset=self.dsname, feature_spaces=self.feature_spaces, 
        instance_space=self.instance_space)
    return FeatureMap(raw, split=self.split, metadata=md)

  def tokenize(self, extractor):
    # TODO: How does this broadcast?
    raise NotImplementedError

class DomainCrossValidation(DataProxy):
  """
  Cross-validate across a set of domains.
  """
  metadata = {}
  def __init__(self, datasets, store=None, feature_spaces=None,
        class_space=None, sequence_name=None, tokenstream_name=None):
    ds, datasets = datasets[0], datasets[1:]
    proxy = DataProxy(ds, store, feature_spaces=feature_spaces,
        class_space=class_space, sequence_name=sequence_name, 
        tokenstream_name=tokenstream_name)
    self.proxies = [ proxy ]

    self.inducer = self.proxies[0].inducer
    self.store = self.proxies[0].store
    
    # Build up a list of proxies, one per dataset
    # They can all share the store and inducer
    for ds in datasets:
      proxy = DataProxy(ds, self.store, self.inducer,
          feature_spaces=feature_spaces, class_space=class_space, 
          sequence_name=sequence_name, tokenstream_name=tokenstream_name)
      self.proxies.append(proxy)
    # Sort proxies by their dataset name to avoid identifying different
    # orderings as different tasksets
    self.proxies.sort(key=lambda x:x.dsname)

    self.feature_spaces = feature_spaces
    self.class_space = class_space
    self.sequence_name = sequence_name
    self.tokenstream_name = tokenstream_name

  @property
  def dsname(self):
    return '+'.join(p.dsname for p in self.proxies)

  @property
  def feature_spaces(self):
    return self._feature_spaces

  @feature_spaces.setter
  def feature_spaces(self, value):
    for p in self.proxies:
      p.feature_spaces = value
    self._feature_spaces = as_set(value)

  @property
  def featurelabels(self):
    for p in self.proxies:
      self.inducer.process(p.dataset, fms=self.feature_spaces)
    labels = []
    for feature_space in sorted(self.feature_spaces):
      labels.extend(self.store.get_Space(feature_space))
    return labels

  @property
  def class_space(self):
    return self._class_space

  @property
  def classlabels(self):
    for p in self.proxies:
      self.inducer.process(p.dataset, cms=self.class_space)
    return self.store.get_Space(self.class_space)

  @class_space.setter
  def class_space(self, value):
    for p in self.proxies:
      p.class_space = value
    self._class_space = value

  @property
  def split_name(self):
    return 'DomainCrossValidation'

  @property
  def split(self):
    """
    Leave-one-out cross-validation of domains
    """
    num_domains = len(self.proxies)
    num_inst = sum(len(p.instancelabels) for p in self.proxies)

    start_index = [ 0 ]
    for p in self.proxies:
      start_index.append( start_index[-1] + len(p.instancelabels) )

    retval = numpy.zeros((num_inst,num_domains,2), dtype=bool)
    for i in xrange(num_domains):
      retval[start_index[i]:start_index[i+1],i,1] = True # Set Eval
      retval[:,i,0] = numpy.logical_not(retval[:,i,1]) #Train on all that are not eval
    return retval

  @property
  def tokenstream_name(self):
    return self._tokenstream_name

  @tokenstream_name.setter
  def tokenstream_name(self, value):
    for p in self.proxies:
      p.tokenstream_name = value
    self._tokenstream_name = value

  @property
  def tokenstream(self):
    ts = []
    for p in self.proxies:
      ts.extend(p.tokenstream)
    return ts

  @property
  def instance_space(self):
    """ Returns a concatenation of the two instance spaces """
    return '+'.join(p.instance_space for p in self.proxies)

  @property
  def instancelabels(self):
    # TODO: May need to handle clashes in labels. Could prefix dataset name.
    labels = []
    for p in self.proxies:
      labels.extend(p.instancelabels)
    return labels

  @property
  def classmap(self):
    cms = [ p.classmap.raw for p in self.proxies ]
    raw = numpy.vstack(cms)
    md = dict(dataset=self.dsname, class_space=self.class_space, 
        instance_space=self.instance_space)
    return ClassMap(raw, split=self.split, metadata=md)

  @property
  def featuremap(self):
    # NOTE: We access the featurelabels in order to ensure that
    # full common feature space is learned before we attempt to access
    # the actual featuremaps
    for p in self.proxies:
      p.featurelabels

    fms = [ p.featuremap.raw for p in self.proxies ]
    raw = scipy.sparse.vstack(fms).tocsr()
    md = dict(dataset=self.dsname, feature_spaces=self.feature_spaces, 
        instance_space=self.instance_space)
    return FeatureMap(raw, split=self.split, metadata=md)

  def tokenize(self, extractor):
    # TODO: How does this broadcast?
    raise NotImplementedError


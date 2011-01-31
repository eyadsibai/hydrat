import logging
import os
import numpy

import hydrat
from hydrat.common.pb import ProgressIter
from hydrat.store import Store, StoreError, NoData, AlreadyHaveData
from hydrat.preprocessor.model.inducer.dataset import DatasetInducer
from hydrat.preprocessor.features.transform import union
from hydrat.task.sampler import membership_vector
from hydrat.common import as_set
logger = logging.getLogger(__name__)
from hydrat.common.decorators import deprecated

@deprecated
def init_workdir(path, newdirs=["models","tasks","results","output"]):
  """ Initialize the working directory, where various intermediate files will be stored.
  This is not to be considered a scratch folder, since the files stored here can be re-used.
  @param path The path to initialize
  """
  if os.path.exists(path):
    logger.warning('%s already exists', path)
  else:
    os.makedirs(path)
    for dir in newdirs:
      os.mkdir(os.path.join(path,dir))

class Framework(object):
  def __init__( self
              , dataset
              , store = None
              ):
    self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
    self.notify('Initializing')
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

    self.feature_spaces = None
    self.feature_desc = None
    self.class_space = None
    self.learner = None
    self.split_name = None
    self.interpreter = None
    # NOTE: The following is a hack to avoid a particular issue resulting from a bad interaction
    # between pytables and h5py. Pytables registers an exitfunc with atexit which closes any
    # open h5files by calling their .close() method. This happens before any __del__ methods
    # are invoked. When h5py is imported, this causes an error. However, explicitly closing a 
    # pytables tableFile with its close method does not cause the same error. This hook forces
    # that to happen before pytables' atexit is called. It calls store.__del__ to avoid diving
    # too deeply into the store implementation. 
    # Perhaps the issue is with h5py registering a hook that gets called first. The hooks are 
    # called LIFO, so ours will always get called first.
    import atexit; atexit.register(lambda: self.store.__del__())
  
  @property
  def train_indices(self):
    if self.split_name is None:
      # TODO: Find a faster way of computing this if necessary.
      return numpy.ones(len(self.store.get_Space(self.dataset.instance_space)), dtype='bool')
    else:
      return self.split[:,0,0]

  @property
  def featuremap(self):
    self.inducer.process_Dataset(self.dataset, fms=self.feature_spaces)
    ds_name = self.dataset.__name__
    featuremaps = []
    for feature_space in sorted(self.feature_spaces):
      featuremaps.append(self.store.get_FeatureMap(ds_name, feature_space))

    # Join the featuremaps into a single featuremap
    fm = union(*featuremaps)
    # NOTE: caution here, sparse arrays must not be indexed by boolean arrays
    return fm[numpy.flatnonzero(self.train_indices)]

  @property
  def featurelabels(self):
    labels = []
    # TODO: Handle unlabelled (EG transformed) feature spaces
    for feature_space in sorted(self.feature_spaces):
      labels.extend(self.store.get_Space(feature_space))
    return labels

  @property
  def classmap(self):
    self.inducer.process_Dataset(self.dataset, cms=self.class_space)
    ds_name = self.dataset.__name__
    cm = self.store.get_ClassMap(ds_name, self.class_space)
    return cm[self.train_indices]

  @property
  def classlabels(self):
    return self.store.get_Space(self.class_space)

  @property
  def classifier(self):
    learner = self.learner
    if self.learner is None:
      raise ValueError, "Learner has not been set"
    cm = self.classmap.raw
    fm = self.featuremap.raw
    self.notify("Training '%s'" % self.learner)
    classifier = learner(fm, cm)
    return classifier

  @property
  def split(self):
    # TODO: grab from store instead. must ensure it has been induced.
    #       this code goes into the inducer
    split_raw = self.dataset.split(self.split_name)
    if 'train' in split_raw and 'test' in split_raw:
      # Train/test type split.
      all_ids = self.dataset.instance_ids
      train_ids = membership_vector(all_ids, split_raw['train'])
      test_ids = membership_vector(all_ids, split_raw['test'])
      split = numpy.dstack((train_ids, test_ids)).swapaxes(0,1)
    elif any(key.startswith('fold') for key in split_raw):
      # Cross-validation folds
      all_ids = self.dataset.instance_ids
      folds_present = sorted(key for key in split_raw if key.startswith('fold'))
      partitions = []
      for fold in folds_present:
        test_ids = membership_vector(all_ids, split_raw[fold])
        train_docids = sum((split_raw[f] for f in folds_present if f is not fold), [])
        train_ids = membership_vector(all_ids, train_docids)
        partitions.append( numpy.dstack((train_ids, test_ids)).swapaxes(0,1) )
      split = numpy.hstack(partitions)
    else:
      raise ValueError, "Unknown type of split"
    return split

  @property
  def sequence(self):
    # TODO: Does this need to be filtered by training_indices?
    if self.sequence_name is None:
      return None
    else:
      return self.store.get_Sequence(self.dataset.__name__, self.sequence_name)

  def notify(self, str):
    self.logger.info(str)

  def set_feature_spaces(self, feature_spaces):
    self.inducer.process_Dataset( self.dataset, fms = feature_spaces)
    self.feature_spaces = as_set(feature_spaces)
    self.feature_desc = tuple(sorted(self.feature_spaces))
    self.notify("Set feature_spaces to '%s'" % str(feature_spaces))
    self.configure()

  def set_class_space(self, class_space):
    self.inducer.process_Dataset( self.dataset, cms = class_space)
    self.class_space = class_space
    self.notify("Set class_space to '%s'" % class_space)
    self.configure()

  def set_learner(self, learner):
    self.learner = learner
    self.notify("Set learner to '%s'" % learner)
    self.configure()

  def set_split(self, split):
    """
    Setting a split causes the framework to only use the 'train' portion of the split
    for training. Setting a split that does not contain a 'train' partition will cause
    an error.
    """
    self.split_name = split
    self.notify("Set split to '%s'" % split)
    self.configure()

  def set_interpreter(self, interpreter):
    self.interpreter = interpreter
    self.notify("Set interpreter to '%s'" % interpreter)
    self.configure()

  def set_sequence(self, sequence):
    self.inducer.process_Dataset( self.dataset, sqs = sequence)
    self.sequence_name = sequence
    self.notify("Set sequence to '%s'" % sequence)
    self.configure()

  def configure(self): 
    self.inducer.process_Dataset(self.dataset, fms=self.feature_spaces, cms=self.class_space)

  def process_tokenstream(self, tsname, extractor):
    dsname = self.dataset.__name__
    # Definition of space name.
    space_name = '_'.join((tsname,extractor.__name__))
    if not self.store.has_Data(dsname, space_name):
      self.notify("Inducing TokenStream '%s'" % tsname)
      # We always call this as if the ts has already been processed it is a fairly 
      # cheap no-op
      self.inducer.process_Dataset(self.dataset, tss=tsname)

      self.notify("Reading TokenStream '%s'" % tsname)
      tss = self.store.get_TokenStreams(dsname, tsname)
      instance_ids = self.store.get_InstanceIds(dsname)
      feat_dict = dict()
      for i, id in enumerate(ProgressIter(instance_ids, 'Processing %s' % extractor.__name__)):
        feat_dict[id] = extractor(tss[i])
      self.inducer.add_Featuremap(dsname, space_name, feat_dict)

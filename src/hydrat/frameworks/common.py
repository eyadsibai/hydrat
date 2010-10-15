import logging
import os

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
  
  @property
  def featuremap(self):
    self.inducer.process_Dataset(self.dataset, fms=self.feature_spaces)
    ds_name = self.dataset.__name__
    featuremaps = []
    for feature_space in sorted(self.feature_spaces):
      featuremaps.append(self.store.get_FeatureMap(ds_name, feature_space))

    # Join the featuremaps into a single featuremap
    fm = union(*featuremaps)
    return fm

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
    return self.store.get_ClassMap(ds_name, self.class_space)

  @property
  def classlabels(self):
    return self.store.get_Space(self.class_space)

  @property
  def classifier(self):
    learner = self.learner
    if self.learner is None:
      raise ValueError, "Learner has not been set"
    if self.split_name is None:
      cm = self.classmap.raw
      fm = self.featuremap.raw
    else:
      mv = membership_vector(self.dataset.instance_ids, self.split['train'])
      train_indices = mv.nonzero()[0]
      cm = self.classmap.raw[train_indices]
      fm = self.featuremap.raw[train_indices]
    self.notify("Training '%s'" % self.learner)
    classifier = learner(fm, cm)
    return classifier

  @property
  def split(self):
    # TODO: Obtain this from store once splits are stored as well
    return self.dataset.split(split)

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

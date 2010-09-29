import logging
import os

import hydrat
from hydrat.common.pb import ProgressIter
from hydrat.store import Store, StoreError, NoData, AlreadyHaveData
from hydrat.preprocessor.model.inducer.dataset import DatasetInducer
from hydrat.preprocessor.features.transform import union
from hydrat.common import as_set
logger = logging.getLogger(__name__)

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
              , work_path = None
              ):
    self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
    self.notify('Initializing')
    self.dataset = dataset
    self.work_path = work_path

    if work_path is None:
      generic_work_path = hydrat.config.get('paths','work')
      self.work_path = os.path.join(generic_work_path, self.__class__.__name__, dataset.__name__)
    
    init_workdir(self.work_path, ["output"])
    self.outputP  = os.path.join(self.work_path, 'output')
    self.store = Store(os.path.join(self.work_path,'store.h5'), 'a')
    self.inducer = DatasetInducer(self.store)

    self.feature_spaces = None
    self.class_space = None
    self.learner = None
  
  @property
  def featuremap(self):
    ds_name = self.dataset.__name__
    featuremaps = []
    for feature_space in sorted(self.feature_spaces):
      featuremaps.append(self.store.get_Data(ds_name, {'type':'feature','name':feature_space}))

    # Join the featuremaps into a single featuremap
    fm = union(*featuremaps)
    return fm

  @property
  def classmap(self):
    ds_name = self.dataset.__name__
    return self.store.get_Data(ds_name, {'type':'class', 'name':self.class_space})

  @property
  def classlabels(self):
    return self.store.get_Space(self.class_space)

  @property
  def classifier(self):
    cm = self.classmap
    fm = self.featuremap
    self.notify("Training Classifier")
    classifier = self.learner(fm.raw, cm.raw)
    return classifier

  def notify(self, str):
    self.logger.info(str)

  def set_feature_spaces(self, feature_spaces):
    self.inducer.process_Dataset( self.dataset, fms = feature_spaces)
    self.feature_spaces = as_set(feature_spaces)
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

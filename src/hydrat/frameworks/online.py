# Online classification framework.
# TODO: Refactor out the commonalities with Framework.
# TODO: Support feature selection and weighting

import os
import logging
import numpy
import scipy.sparse

import hydrat
import hydrat.common.extractors as ext
from hydrat.store import Store, StoreError, NoData, AlreadyHaveData
from hydrat.preprocessor.model.inducer.dataset import DatasetInducer
from hydrat.frameworks import init_workdir
from hydrat.common import as_set
from hydrat.preprocessor.features.transform import union
from hydrat.preprocessor.model.inducer import invert_text

class OnlineFramework(object):
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
    self.classifier = None
    self.vectorize = None
    self.classlabels = None

  def notify(self, str):
    self.logger.info(str)

  def set_feature_spaces(self, feature_spaces):
    # Sanity check - we must know how to derive the tokenstream required.
    # For now, we only handle byte tokenstream.
    self.feature_spaces = as_set(feature_spaces)
    # TODO: Make this work with other tokenstreams. Need to check if we know how to transform
    #       text into the expected tokenstream.
    if not all(f.startswith('byte') for f in self.feature_spaces):
      raise ValueError, "can only handle byte-derived feature spaces"
    self.notify("Set feature_spaces to '%s'" % str(feature_spaces))
    self.configure()

  def set_class_space(self, class_space):
    self.class_space = class_space
    self.notify("Set class_space to '%s'" % class_space)
    self.configure()

  def set_learner(self, learner):
    self.learner = learner
    self.notify("Set learner to '%s'" % learner)
    self.configure()

  def is_configurable(self):
    # Modified from Framework
    return self.feature_spaces is not None\
       and self.class_space is not None\
       and self.learner is not None

  def configure(self):
    if self.is_configurable():
      # Ensure that the relevant feature/class spaces have been modelled.
      self.inducer.process_Dataset(self.dataset, fms=self.feature_spaces, cms=self.class_space)
      s = self.store
      ds_name = self.dataset.__name__
      cm = s.get_Data(ds_name, dict(type='class', name=self.class_space)) 

      featuremaps = []
      for feature_space in sorted(self.feature_spaces):
        featuremaps.append(self.store.get_Data(ds_name, {'type':'feature','name':feature_space}))

      # Join the featuremaps into a single featuremap
      fm = union(*featuremaps)

      # Instantiate classifier
      self.classifier = self.learner(fm.raw, cm.raw)

      # Set our classlabels for later decoding classifier output
      self.classlabels = numpy.array(s.get_Space(self.class_space))

      # Set up feature extraction machinery for incoming text.
      fns = {}
      fis = {}
      exs = {}
      for feature_space in self.feature_spaces:
        feat_names = s.get_Space(feature_space)
        fns[feature_space] = feat_names
        fis[feature_space] = dict( (k,v) for v, k in enumerate(feat_names) )
        # TODO: Improve the sophistication of this.
        #       In particular, need to be able to work with extractors 
        #       declared by the user.
        #       This is also where we should handle tokenstream transforms
        exs[feature_space] = getattr(ext, feature_space.split('_')[1])

      def vectorize(text):
        batch_size = len(text)
        fms = []
        for feature_space in sorted(self.feature_spaces):
          feat_names = fns[feature_space]
          feat_index = fis[feature_space] 
          extractor  = exs[feature_space]

          feat_map = scipy.sparse.dok_matrix((batch_size, len(feat_names)))
          for i, t in enumerate(text):
            inv_text = extractor(t)
            for feat in inv_text:
              if feat in feat_index:
                j = feat_index[feat]
                feat_map[i,j] = inv_text[feat]
          fms.append( feat_map )
        return scipy.sparse.hstack(fms).tocsr()
      
      self.vectorize = vectorize

      
  def classify(self, text):
    # classify a text instance
    # need to generate the corresponding features first
    solo = False # Was only one instance supplied to classify?
    if isinstance(text, str):
      text = [text]
      solo = True
    elif isinstance(text, unicode):
      raise ValueError, "Can only handle byte streams for now"
    feat_map = self.vectorize(text)
    result = self.classifier(feat_map)
    if solo:
      return self.classlabels[result[0]]
    else:
      return [self.classlabels[r] for r in result]

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
      for i, id in enumerate(ProgressIter(instance_ids, 'Processing TokenStream')):
        feat_dict[id] = extractor(tss[i])
      self.inducer.add_Featuremap(dsname, space_name, feat_dict)


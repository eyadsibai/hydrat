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
from hydrat.common import as_set
from hydrat.preprocessor.features.transform import union
from hydrat.preprocessor.model.inducer import invert_text
from hydrat.frameworks.common import Framework

class OnlineFramework(Framework):
  def __init__( self
              , dataset
              , work_path = None
              ):
    Framework.__init__(self, dataset, work_path)

    self.vectorize = None
    self.__classifier = None

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

  def is_configurable(self):
    return self.feature_spaces is not None\
       and self.class_space is not None\
       and self.learner is not None

  def configure(self):
    if self.is_configurable():
      # Ensure that the relevant feature/class spaces have been modelled.
      self.inducer.process_Dataset(self.dataset, fms=self.feature_spaces, cms=self.class_space)

      # Set up feature extraction machinery for incoming text.
      fns = {}
      fis = {}
      exs = {}
      for feature_space in self.feature_spaces:
        feat_names = self.store.get_Space(feature_space)
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
      self.__classifier = self.classifier

      
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
    result = self.__classifier(feat_map)
    classlabels = numpy.array(self.classlabels)
    if solo:
      return classlabels[result[0]]
    else:
      return [classlabels[r] for r in result]

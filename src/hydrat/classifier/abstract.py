"""
This module specifies the abstract interface that all classifier modules should
implement, and also provides some convenience methods.
"""
import numpy as n
import logging
import time
class ClassifierError(Exception): pass
class NoClassLabelsError(ClassifierError): pass
class NoFeaturesError(ClassifierError): pass
class NotInstalledError(ClassifierError): pass

class Learner(object):
  def __init__(self):
    self.logger = logging.getLogger("hydrat.classifier.learner.%s"%self.__name__)
    self._check_installed()

  def __call__(self, feature_map, class_map):
    num_docs, num_classes = class_map.shape
    num_features = feature_map.shape[1]
    self.logger.debug\
      ( "learning on %d documents in %d classes with %d features"
      , num_docs
      , num_classes
      , num_features
      ) 
    start = time.time()
    classifier = self._learn(feature_map, class_map)
    timetaken = time.time() - start
    classifier.metadata['learner']          = self.__name__
    classifier.metadata['learner_params']   = self.params
    classifier.metadata['learn_time']       = timetaken
    classifier.metadata['train_feat_count'] = num_features
      
    self.logger.debug("learning took %.1f seconds", timetaken)
    return classifier

  def _check_installed(self):
    """ Check that any external tools required are actually installed 
    Should raise an exception if they are not, and not return anything if they are
    """
    self.logger.warning("Learner '%s' does not implement _check_installed", self.__name__)

  @property
  def params(self):
    try:
      return self._params()
    except NotImplementedError:
      self.logger.warning("Learner '%s' does not implement _params", self.__name__)
      return None

  @property
  def desc(self):
    return self.__name__, self._params()

  def _learn(self, feature_map, class_map):
    """ Train a classifier
        Returns a Classifier object
    """
    raise NotImplementedError

  def _params(self):
    """
    Returns a dictionary describing the learner
    Ideally should be able to pass this to the learner class' __init__ **kwargs
    """
    raise NotImplementedError


class Classifier(object):
  def __init__(self):
    self.logger   = logging.getLogger("hydrat.classifier.%s"%self.__name__)
    self.metadata = { 'classifier' : self.__name__ }

  def __call__(self, feature_map):
    self.logger.debug("classifying %d documents", feature_map.shape[0])
    cl_num_feats = feature_map.shape[1]
    tr_num_feats = self.metadata['train_feat_count']
    if cl_num_feats != tr_num_feats: 
      raise ValueError, "Trained on %d features, %d for classification" % ( tr_num_feats
                                                                          , cl_num_feats
                                                                          )

    # Check that we have not provided any empty instances
    for i,row in enumerate(feature_map):
      if len(row.nonzero()) == 0:
        self.logger.warning("Empty classification instance at index %d!", i)
    start                           = time.time()
    classifications                 = self._classify(feature_map)
    timetaken                       = time.time() - start
    self.metadata['classify_time']  = timetaken
    self.logger.debug("classification took %.1f seconds", timetaken)
    return classifications 

  def _classify(self, feature_map):
    """ Classify a set of documents represented as an array of features
        Returns a boolean array:
          axis 0: document index
          axis 1: classlabels
    """
    raise NotImplementedError

class NullLearner(Learner):
  __name__ = "nulllearner"

  def __init__(self, classifier_constructor, name, *args, **kwargs):
    self.__name__ = name 
    Learner.__init__(self)
    assert issubclass(classifier_constructor, LearnerlessClassifier)
    self._args                   = args
    self._kwargs                 = kwargs
    self.classifier_constructor  = classifier_constructor

  def _learn(self, feature_map, class_map ):
    return self.classifier_constructor( self.__name__
                                      , feature_map
                                      , class_map
                                      , *self._args
                                      , **self._kwargs
                                      )

class LearnerlessClassifier(Classifier):
  def __init__(self, name, feature_map, class_map):
    self.__name__ = name
    Classifier.__init__(self)
    self.train_fv  = feature_map
    self.train_cv  = class_map

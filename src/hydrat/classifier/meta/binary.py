# Binarizing Classifier.
# Works by turning an n-of-m task into m binary tasks.
# This is done in a 1 vs all fashion.

import numpy
from hydrat.classifier.abstract import Learner, Classifier
from hydrat.common.pb import ProgressIter

class BinaryLearner(Learner):
  def __init__(self, learner):
    self.__name__ = learner.__name__
    Learner.__init__(self)
    self.learner = learner

  def _params(self):
    params = dict(self.learner.params)
    params['multiclass'] = 'binarized'
    return params

  def _learn(self, feature_map, class_map, **kwargs):
    num_classes = class_map.shape[1]
    classifiers = []
    for cl in ProgressIter(range(num_classes),label='Binary Learn'):
      mask = class_map[:,cl]
      # Build a two-class task
      # The second class is the "True" class
      submap = numpy.vstack((numpy.invert(mask),mask)).transpose()
      classifiers.append(self.learner(feature_map, submap, **kwargs))
    return BinaryClassifier(classifiers, self.learner.__name__)

class BinaryClassifier(Classifier):
  def __init__(self, classifiers, name):
    self.__name__ = name
    Classifier.__init__(self)
    self.classifiers = classifiers

  def _classify(self, feature_map, **kwargs):
    outcomes = []
    for c in ProgressIter(self.classifiers, label="Binary Classify"):
      # We only want the members of the second class
      r = c(feature_map, **kwargs)
      outcomes.append(r[:,1])

    result = numpy.vstack(outcomes).transpose()
    return result

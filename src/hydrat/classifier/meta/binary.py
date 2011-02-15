# Binarizing Classifier.
# Works by turning an n-of-m task into m binary tasks.
# This is done in a 1 vs all fashion.

import numpy
from hydrat.classifier.abstract import Learner, Classifier
from hydrat.common.pb import ProgressIter

def negative_classifier(feature_map, **kwargs):
  retval = numpy.hstack([
    numpy.ones((feature_map.shape[0], 1), dtype='bool'),
    numpy.zeros((feature_map.shape[0], 1), dtype='bool'),
    ])
  return retval

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
      if mask.sum() == 0:
        # There are no exemplars for this class, so we can just slot in a dummy classifier
        # that always returns negative
        classifiers.append(negative_classifier)
      else:
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

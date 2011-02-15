"""
hydrat's interface to scikit.learn

http://scikit-learn.sourceforge.net

Marco Lui <saffsd@gmail.com> October 2010
"""
import numpy

from hydrat.task.sampler import isOneofM
from hydrat.classifier.abstract import Learner, Classifier, NotInstalledError
class ScikitL(Learner):
  """
  Lightweight wrapper for scikit's learner interface
  """
  __name__ = 'scikit'
  def __init__(self, learn_class, **kwargs):
    Learner.__init__(self)
    self.learn_class = learn_class
    self.kwargs = kwargs

  def _params(self):
    md = dict(self.kwargs)
    md['learner'] = self.learn_class.__name__
    return md

  def _learn(self, feature_map, class_map):
    if not isOneofM(class_map):
      raise ValueError, "can only use one-of-m classmaps"
    learner = self.learn_class(**self.kwargs)
    targets = class_map.argmax(axis=1)
    learner.fit(feature_map, targets)
    return ScikitC(learner, class_map.shape[1])

class ScikitC(Classifier):
  __name__ = 'scikits'
  def __init__(self, learner, num_class):
    Classifier.__init__(self)
    self.learner = learner
    self.num_class = num_class

  def _classify(self, feature_map):
    pred = self.learner.predict(feature_map)
    classif = numpy.zeros((feature_map.shape[0], self.num_class), dtype='bool')
    for i,p in enumerate(pred):
      classif[i,p] = True
    return classif

# Convenience methods
from scikits.learn import svm
def SVC(**kwargs): return ScikitL(svm.sparse.SVC, **kwargs)
def NuSVC(**kwargs): return ScikitL(svm.sparse.NuSVC, **kwargs)
def LinearSVC(**kwargs): return ScikitL(svm.sparse.LinearSVC, **kwargs)

# TODO: There are generalized linear models available for sparse features
# TODO: Some of the classifiers are only implemented for dense features, could investigate
#       using them but would need to be careful of very large spaces.
# TODO: Warn if scikits.learn is not installed

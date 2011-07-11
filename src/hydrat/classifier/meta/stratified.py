"""
Based on code by lwang
Adapted by mlui
"""
import numpy
import math
from hydrat.classifier.abstract import Learner, Classifier
from hydrat.result.interpreter import SingleHighestValue
from hydrat.task.sampler import stratify_with_index

class StratifiedLearner(Learner):
  def __init__(self, learner, interpreter=SingleHighestValue()):
    self.__name__ = learner.__name__
    Learner.__init__(self)
    self.learner = learner
    self.interpreter = interpreter
    
  def _params(self):
    params = dict(self.learner.params)
    params['multiclass'] = 'stratified'
    return params

  def _learn(self, feature_map, class_map, **kwargs):
    stratified_class_map, reversed_strata_index = stratify_with_index(class_map)
    strata_index = dict( (v,k) for k,v in reversed_strata_index.items())
    classifier = self.learner(feature_map, stratified_class_map, **kwargs)
    num_classes = class_map.shape[1]
    return StratifiedClassifier(classifier, num_classes, strata_index, self.interpreter, self.__name__)
      
def int2boolarray(number, array_len):
  a = numpy.zeros(array_len, dtype=bool)
  index = array_len
  while number > 0:
    index -= 1
    number, rem = divmod(number, 2)
    if rem != 0:
      a[index] = True
  return a
  
class StratifiedClassifier(Classifier):
  def __init__(self, classifier, num_class, strata_index, interpreter, name):
    self.__name__ = name
    Classifier.__init__(self)
    self.classifier = classifier
    self.interpreter = interpreter
    self.num_class = num_class
    self.strata_index = strata_index

  def _classify(self, feature_map, **kwargs):
    outcome = self.interpreter(self.classifier(feature_map, **kwargs))
    num_instance = feature_map.shape[0]
    num_class = self.num_class
    
    result = []
    for i, row in enumerate(outcome):
      nonzero_indices = numpy.flatnonzero(row)
      if len(nonzero_indices) != 1:
        raise ValueError, "Have more than one nonzero index"
      multi_index = nonzero_indices[0]
      multi_identifier = self.strata_index[multi_index]
      result.append(int2boolarray(multi_identifier, self.num_class))
    return numpy.vstack(result)

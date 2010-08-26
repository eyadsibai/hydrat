import numpy
import logging
from hydrat.common.transform import Transformer
from infogain import ig_bernoulli
from weighting_function import CavnarTrenkle94

class KeepRule(object):
  __name__ = 'keeprule'
  def __call__(self, weight_vector):
    """Returns indices to be kept"""
    raise NotImplemented, "Need to implement the rule!"

class HighestN(KeepRule):
  def __init__(self, n):
    self.n = n
    self.__name__ = 'HighestN%d' % n

  def __call__(self, weight_vector):
    """
    Note that in the case of weight equality this is biased towards
    low-indexed features by nature of numpy's argsort.
    """
    return numpy.argsort(weight_vector)[-self.n:]

class NonZero(KeepRule):
  __name__ = "NonZero"

  def __call__(self, weight_vector):
    return numpy.flatnonzero(weight_vector)

class FeatureSelect(Transformer):
  def __init__(self, weighting_function, keep_rule):
    self.__name__ = weighting_function.__name__
    self.__name__+= '_fs%s' % keep_rule.__name__
    self.logger = logging.getLogger('hydrat.classifier.featureselect.' + self.__name__)
    self.keep_rule = keep_rule
    self.weighting_function = weighting_function
    self.keep_indices = None

  def learn(self, feature_map, class_map):
    self.logger.debug('Learning Weights')
    weights = self.weighting_function(feature_map, class_map)
    keep = self.keep_rule(weights)
    self.keep_indices = keep

  def apply(self, feature_map):
    assert self.keep_indices is not None, "Weights have not been learned!"
    self.logger.debug('Applying Weights')
    
    selected_map = feature_map.transpose()[self.keep_indices].transpose()
    return selected_map 

ig_bern_top500 = FeatureSelect(ig_bernoulli, HighestN(500))
cavnar_trenkle = FeatureSelect(CavnarTrenkle94(300), NonZero())

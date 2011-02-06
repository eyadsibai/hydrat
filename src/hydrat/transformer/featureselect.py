import numpy
from hydrat.transformer import Transformer

class FeatureSelect(Transformer):
  def __init__(self, weighting_function, keep_rule):
    self.__name__ = 'FS-%s-%s' % (weighting_function.__name__, keep_rule.__name__)
    Transformer.__init__(self)
    self.keep_rule = keep_rule
    self.weighting_function = weighting_function
    self.keep_indices = None
    self.weights[weighting_function.__name__] = None

  def learn(self, feature_map, class_map):
    self.logger.debug('Learning Weights')
    wf_name = self.weighting_function.__name__
    if self.weights[wf_name] is None:
      self.weights[wf_name] = self.weighting_function(feature_map, class_map)

    weights = self.weights[wf_name]
    self.keep_indices = self.keep_rule(weights)

  def apply(self, feature_map):
    assert self.keep_indices is not None, "Weights have not been learned!"
    self.logger.debug('Applying Weights')
    
    selected_map = feature_map[:,self.keep_indices]
    return selected_map 

class KeepRule(object):
  # TODO: These are nearly identical to ResultInterpreter. Should refactor them together
  def __init__(self):
    if not hasattr(self, '__name__'):
      self.__name__ = self.__class__.__name__

  def __call__(self, vector):
    """Returns indices to be kept"""
    raise NotImplementedError

def nonzero(vector):  return numpy.flatnonzero(vector)

class HighestN(KeepRule):
  def __init__(self, n):
    self.__name__ = 'HighestN-%d' % n
    KeepRule.__init__(self)
    self.n = n

  def __call__(self, vector):
    """
    Note that in the case of weight equality this is biased towards
    low-indexed features by nature of numpy's argsort.
    """
    return numpy.argsort(vector)[-self.n:]

class Exceeds(KeepRule):
  def __init__(self, n):
    self.__name__ = 'Exceeds-%d' % n
    KeepRule.__init__(self)
    self.n = n

  def __call__(self, weight_vector):
    return numpy.flatnonzero(weight_vector >= self.n)


import hydrat.common.weight as weight 
cavnar_trenkle94 = FeatureSelect(weight.CavnarTrenkle94(300), nonzero)
def term_count_exceeds(x): return FeatureSelect(weight.TermFrequency(), Exceeds(x))
def doc_count_exceeds(x): return FeatureSelect(weight.DocumentFrequency(), Exceeds(x))
def ig_bern(x): return FeatureSelect(weight.ig_bernoulli, HighestN(x))

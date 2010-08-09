import logging
import numpy
from hydrat.common import entropy
from hydrat.common.transform.weighting_function import WeightingFunction

def bernoulli(v):
  nonzero = v.nonzero()[0]
  zero = numpy.array(list(set(range(v.shape[0])) - set(nonzero)))
  return [zero, nonzero]

class UniformBand(object):
  def __init__(self, bands):
    self.__name__ = 'uniform%dband' % bands
    self.bands = bands

  def __call__(self, v):
    limit = float(numpy.max(v.data) + 1)
    bins = numpy.digitize(v, numpy.arange(0, limit, limit/self.bands))
    r = numpy.empty((self.bands, len(v)), dtype=bool)
    for i in range(self.bands):
      r[i] = (bins == (i+1))
    return r

class EquisizeBand(object):
  def __init__(self, bands):
    self.__name__ = 'equisize%dband' % bands
    self.bands = bands

  def __call__(self, v):
    r = numpy.empty((self.bands, v.shape[0]), dtype=bool)
    band_size = 100.0 / (self.bands)
    for i in range(self.bands):
      r[i] = numpy.logical_and( (i * band_size) <= v, v < (i * (band_size + 1)) )
    return r
 
class InfoGain(WeightingFunction):
  __name__ = 'infogain'
  def __init__(self, feature_discretizer):
    self.__name__ = 'infogain_' + feature_discretizer.__name__
    self.feature_discretizer = feature_discretizer
    self.logger = logging.getLogger('hydrat.common.transform.infogain')

  def weight(self, feature_map, class_map):
    overall_class_distribution = class_map.sum(axis=0)
    total_instances = float(feature_map.shape[0])
    
    # Calculate  the entropy of the class distribution over all instances 
    H_P = entropy(overall_class_distribution)
    self.logger.info("Overall entropy: %.2f", H_P)
      
    feature_weights = numpy.zeros(feature_map.shape[1], dtype=float)
    for i in range(len(feature_weights)):
      H_i = 0.0
      for f_mask in self.feature_discretizer(feature_map[:,i]):
        f_count = len(f_mask) 
        if f_count == 0: continue # Skip partition if no instances are in it
        f_distribution = class_map[f_mask].sum(axis=0)
        f_entropy = entropy(f_distribution)
        f_weight = f_count / total_instances
        H_i += f_weight * f_entropy

      feature_weights[i] =  H_P - H_i

    #import pdb;pdb.set_trace()
    return feature_weights

ig_bernoulli = InfoGain(bernoulli)
ig_uniform5band = InfoGain(UniformBand(5))
ig_equisize5band = InfoGain(EquisizeBand(5))

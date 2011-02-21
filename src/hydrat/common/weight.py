import numpy
import logging
from hydrat.common import entropy, rankdata
from hydrat.common.pb import ProgressIter

class WeightingFunction(object):
  """
  Class representing a weighting function, which is simply
  a single value for each term (feature).
  must implement self.weight(feature_map, class_map)
  """
  def __init__(self):
    if not hasattr(self, '__name__'):
      self.__name__ = self.__class__.__name__
    self.logger = logging.getLogger(__name__ + '.' + self.__name__)

  def __call__(self, feature_map, class_map):
    return self.weight(feature_map, class_map)

  def weight(self, feature_map, class_map):
    raise NotImplementedError

class TermFrequency(WeightingFunction):
  """
  Returns the summation across all instances
  """
  def weight(self, feature_map, class_map):
    raw = feature_map.sum(axis=0)
    return numpy.array(raw)[0]

class DocumentFrequency(WeightingFunction):
  """
  Returns how many instances each term occurs more than threshold
  times in.
  """
  def __init__(self, threshold = 0):
    WeightingFunction.__init__(self)
    self.threshold = threshold

  def weight(self, feature_map, class_map):
    fm = feature_map.copy()
    fm.data = (fm.data > self.threshold)
    raw = fm.sum(axis=0)
    return numpy.array(raw)[0]

class CavnarTrenkle94(WeightingFunction):
  """
  Weighting function generalized from the highly 
  influential 1994 paper N-gram based text categorization

  The basic principle is to keep the most common N features
  from each class. To generalize this, we compute a score
  for each feature. For a given N, a feature should be
  kept if its score is < N. 

  This is fairly simple to compute. For each class, features
  are ranked on the basis of their frequency. The score for
  each feature is then the minimum rank across all classes.

  TODO: Generalize this approach to use a per-class metric
  other than simple counts. For example, could use mutual
  information. Would need some way to work with saved weights.
  """
  def weight(self, feature_map, class_map):
    profiles = []
    for cl_i in ProgressIter(range(class_map.shape[1]), 'CavnarTrenkle94'):
      # Get the instance indices which correspond to this class
      class_indices = numpy.flatnonzero(class_map[:,cl_i])
      if len(class_indices) == 0: continue # Skip this class: no instances
      # Sum features over all instances in the class
      class_profile = numpy.array(feature_map[class_indices].sum(axis=0))[0]
      # Add this profile to the list
      profiles.append(rankdata(class_profile))
    # The feature weights are the minimum across all classes
    feature_weights = numpy.min(profiles, axis=0)
    return feature_weights

class InfoGain(WeightingFunction):
  def __init__(self, feature_discretizer):
    self.__name__ = 'infogain-' + feature_discretizer.__name__
    WeightingFunction.__init__(self)
    self.feature_discretizer = feature_discretizer
  
  def weight(self, feature_map, class_map):
    overall_class_distribution = class_map.sum(axis=0)
    total_instances = float(feature_map.shape[0])
    
    # Calculate  the entropy of the class distribution over all instances 
    H_P = entropy(overall_class_distribution)
    self.logger.debug("Overall entropy: %.2f", H_P)
      
    feature_weights = numpy.zeros(feature_map.shape[1], dtype=float)
    for i in ProgressIter(range(len(feature_weights)), 'InfoGain'):
      H_i = 0.0
      for f_mask in self.feature_discretizer(feature_map[:,i]):
        f_count = len(f_mask) 
        if f_count == 0: continue # Skip unused features
        f_distribution = class_map[f_mask].sum(axis=0)
        f_entropy = entropy(f_distribution)
        f_weight = f_count / total_instances
        H_i += f_weight * f_entropy

      feature_weights[i] =  H_P - H_i

    return feature_weights

from discretize import bernoulli, UniformBand, EquisizeBand
ig_bernoulli = InfoGain(bernoulli)
ig_uniform5band = InfoGain(UniformBand(5))
ig_equisize5band = InfoGain(EquisizeBand(5))

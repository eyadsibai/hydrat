import numpy

class WeightingFunction(object):
  """
  Class representing a weighting function, which is simply
  a single value for each term (feature).
  must implement self.weight(feature_map, class_map)
  """
  def __init__(self):
    if not hasattr(self, '__name__'):
      self.__name__ = self.__class__.__name__

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
    raw = (feature_map > self.threshold).sum(axis=0)
    return numpy.array(raw)[0]

class CavnarTrenkle94(WeightingFunction):
  """
  Weighting function generalized from the highly 
  influential 1994 paper N-gram based text categorization
  """
  def __init__(self, count=300):
    WeightingFunction.__init__(self)
    self.count = count
    self.__name__ = 'CavTrenk%d' % count

  def weight(self, feature_map, class_map):
    """
    We return a boolean vector of weights,
    which corresponds to whether to keep the
    feature or not. This should be used with
    a NonZero KeepRule 

    The exact number of features kept will depend on the
    class labels
    """
    feature_weights = numpy.zeros(feature_map.shape[1], dtype=bool)
    for cl_i in range(class_map.shape[1]):
      # Get the instance indices which correspond to this class
      class_indices = numpy.flatnonzero(class_map[:,cl_i])
      # Sum features over all instances in the class
      class_profile = feature_map[class_indices].sum(axis=0)
      # Select the top 'count' indices to keep
      keep_indices = numpy.array(class_profile.argsort())[0][-self.count:]
      # Flag these features
      feature_weights[keep_indices] = True
    return feature_weights

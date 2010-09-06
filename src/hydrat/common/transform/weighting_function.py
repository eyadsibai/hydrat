import numpy

class WeightingFunction(object):
  """
  Class representing a weighting function
  must implement self.weight(feature_map, class_map)
  """
  def __call__(self, feature_map, class_map):
    return self.weight(feature_map, class_map)

  def weight(self, feature_map, class_map):
    raise NotImplementedError

class CavnarTrenkle94(WeightingFunction):
  """
  Weighting function generalized from the highly 
  influential 1994 paper N-gram based text categorization
  """
  def __init__(self, count=300):
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

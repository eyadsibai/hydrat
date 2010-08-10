import logging
import numpy

from scipy.sparse import csr_matrix, lil_matrix

from hydrat.common.transform import Transformer, LearnlessTransformer

### Unsupervised weighting approaches ###
class Discretize(LearnlessTransformer):
  def __init__(self, coefficient = 1000):
    self.__name__ = 'discretize%d' % coefficient
    LearnlessTransformer.__init__(self)
    self.coefficient = coefficient

  def apply(self, feature_map):
    self.logger.info('Discretizing!')
    r = (feature_map * self.coefficient).astype(int)
    r.eliminate_zeros()
    return r

class TFIDF(LearnlessTransformer):
  def __init__(self):
    self.__name__ = 'tfidf'
    LearnlessTransformer.__init__(self)

  def apply(self, feature_map):
    weighted_fm = lil_matrix(fm.raw.shape, dtype=float)
    instance_sizes = fm.raw.sum(axis=1)
    
    # Frequency of each term is the sum alog axis 0
    tf = fm.raw.sum(axis=0)
    # Total number of terms in fm
    total_terms = tf.sum()
    
    #IDF for each term
    idf = numpy.zeros(fm.raw.shape[1])
    for f in fm.raw.nonzero()[1]:
      idf[f] += 1
              
    for i,instance in enumerate(fm.raw):
      size = instance_sizes[i]
      # For each term in the instance
      for j in instance.nonzero()[1]: 
        v = fm.raw[i, j] 
        term_freq =  float(v) / float(size) #TF        
        weighted_fm[i, j] = term_freq * idf[j] #W_{d,t}
    
    return weighted_fm.tocsr()

### Supervised Weighting Approaches ###
##### Infrastructure #####
class Weighter(Transformer):
  __name__ = "weighter"

  def __init__(self):
    self.logger = logging.getLogger('hydrat.classifier.weighter.' + self.__name__)


class SimpleWeighter(Weighter):
  def __init__(self, weighting_function):
    self.__name__ = weighting_function.__name__
    Weighter.__init__(self)
    self.weighting_function = weighting_function
    self.weights = None

  def learn(self, feature_map, class_map):
    self.weights = self.weighting_function(feature_map, class_map)
    
  def apply(self, feature_map):
    assert self.weights is not None, "Weights have not been learned!"
    assert feature_map.shape[1] == len(self.weights), "Shape of feature map is wrong!"

    weighted_feature_map = numpy.empty(feature_map.shape, dtype=float)
    for i,row in enumerate(feature_map):
      weighted_feature_map[i] = row.multiply(self.weights.reshape(row.shape))
    return csr_matrix(weighted_feature_map)

class CutoffWeighter(SimpleWeighter):
  """ Similar to SimpleWeighter, but applies a Cutoff value after doing the weighting
  """
  def __init__(self, weighting_function, threshold = 1):
    SimpleWeighter.__init__(self, weighting_function)
    self.__name__ += '_t%s' % str(threshold)
    self.threshold = threshold
    assert threshold > 0, "Not able to deal with subzero thresholds due to sparse data"
    Weighter.__init__(self)

  def apply(self, feature_map):
    mask = feature_map.toarray() >= self.threshold
    weighted_feature_map = SimpleWeighter.apply(self, feature_map)
    weighted_feature_map = weighted_feature_map.toarray() * mask
    return csr_matrix(weighted_feature_map)


import numpy
from hydrat.transformer import Transformer
from hydrat.common import rankdata

class FeatureSelect(Transformer):
  def __init__(self, weighting_function, keep_rule):
    self.__name__ = 'FS-%s-%s' % (weighting_function.__name__, keep_rule.__name__)
    Transformer.__init__(self)
    self.keep_rule = keep_rule
    self.weighting_function = weighting_function
    self.keep_indices = None
    self.weights[weighting_function.__name__] = None

  def learn(self, feature_map, class_map):
    wf_name = self.weighting_function.__name__
    if wf_name not in self.weights or self.weights[wf_name] is None:
      self.logger.debug('Learning Weights')
      self.weights[wf_name] = self.weighting_function(feature_map, class_map)
    else:
      self.logger.debug('Using learned weights')

    weights = self.weights[wf_name]
    if len(weights) != feature_map.shape[1]:
      raise ValueError, "weight length mismatch"

    self.keep_indices = self.keep_rule(weights)

  def apply(self, feature_map):
    assert self.keep_indices is not None, "Weights have not been learned!"
    self.logger.debug('Applying Weights')
    
    selected_map = feature_map[:,self.keep_indices]
    return selected_map 

class LangDomain(FeatureSelect):
  """
  Implementation of LD feature weighting. See
  
  Cross-domain Feature Selection for Language Identification, Marco Lui, Timothy Baldwin (2011) 
  In Proceedings of the 5th International Joint Conference on Natural Language Processing (IJCNLP 2011), Chiang Mai, Thailand. 
  """
  def __init__(self, domain_map, num_feat = 400):
    """
    domain_map is a 2d boolean array (instances, domains)
    It is used for the IG computation for domain.
    """
    Transformer.__init__(self)
    self.domain_map = domain_map
    self.keep_rule = LessThan(num_feat)
    self.keep_indices = None

  def learn(self, feature_map, class_map, indices):
    """
    Note: we don't keep the LD rankings cached, because they are not absolute.
    The depend on the exact feature set, and thus will vary under composition
    with other feature sets.
    """
    reduced_dm = self.domain_map[indices]
    if 'ig_domain' not in self.weights:
      # IG over all domains
      self.weights['ig_domain'] = weight.ig_bernoulli(feature_map, reduced_dm)
    d_w = self.weights['ig_domain']

    cl_prof = []
    for cl in range(class_map.shape[1]):
      # binarized IG over classes
      cl_id = 'ig_cl{0}'.format(cl)
      if cl_id not in self.weights:
        pos = class_map[:,cl]
        reduced_cm = numpy.hstack((numpy.logical_not(pos)[:,None], pos[:,None]))
        self.weights[cl_id] = weight.ig_bernoulli(feature_map, reduced_cm)

      cl_w = self.weights[cl_id]

      cl_ld_w = cl_w - d_w
      cl_ld_r = rankdata(cl_ld_w, reverse=True)
      cl_prof.append(cl_ld_r)

    ld_w = numpy.min(cl_prof, axis=0)
    self.keep_indices = self.keep_rule(ld_w)


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

class LessThan(KeepRule):
  def __init__(self, n):
    self.__name__ = 'LessThan-%d' % n
    KeepRule.__init__(self)
    self.n = n

  def __call__(self, weight_vector):
    return numpy.flatnonzero(weight_vector < self.n)


import hydrat.common.weight as weight 
cavnar_trenkle94 = FeatureSelect(weight.CavnarTrenkle94(), LessThan(300))
def cavnar_trenkle(x): return FeatureSelect(weight.CavnarTrenkle94(), LessThan(x))
def term_count_exceeds(x): return FeatureSelect(weight.TermFrequency(), Exceeds(x))
def doc_count_exceeds(x): return FeatureSelect(weight.DocumentFrequency(), Exceeds(x))
def term_count_top(x): return FeatureSelect(weight.TermFrequency(), HighestN(x))
def doc_count_top(x): return FeatureSelect(weight.DocumentFrequency(), HighestN(x))
def ig_bern(x): return FeatureSelect(weight.ig_bernoulli, HighestN(x))

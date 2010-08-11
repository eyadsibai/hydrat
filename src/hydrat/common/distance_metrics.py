import numpy 
import logging
from itertools import izip
from numpy import seterr
from numpy import concatenate
from numpy import logical_xor
from numpy.linalg import norm
from hydrat.common import progress, timed_report

def dot(q,r):
  return (q * r.T)[0,0]

seterr(all='raise')

#__all__ = ['dm_cosine','dm_kl','dm_skew','dm_tau']
# Tau has been removed due to biopython not liking numpy.uint64
# TODO: Review Tau for usability!!
__all__ = ['dm_cosine','dm_skew']

class MetricError(Exception): pass
class NullVector(MetricError):
  def __init__(self, vector_num, index):
    MetricError.__init__( self
                        , "Null Vector in vector %d at index %d"%(vector_num, index)
                        )
    self.vector_num = vector_num
    self.index = index

class distance_metric(object):
  """Abstract parent for distance metric classes"""
  __name__ = "distance_metric"
  def __init__(self):
    self.logger = logging.getLogger('hydrat.classifier.distance_metric.' + self.__name__)

  def vector_distances(self, v1, v2):
    raise NotImplementedError

  @property
  def params(self):
    return self._params()

  def _params(self):
    return dict(name=self.__name__)

class dm_cosine(distance_metric):
  """
  Cosine Distance. 
  Note that this DOES NOT return the raw cosine of the angle
  between two vectors. It returns 1-cos(p,q), so that we remain consistent
  in terms of a zero-better approach to distances.
  """
  __name__ = "cosine"

  def vector_distances(self, v1, v2):
    self.logger.debug('Creating dense representation')
    orig_feats = v1.shape[1]
    # Determine shared features
    f1 = numpy.asarray( v1.sum(0) ) [0] > 0
    f2 = numpy.asarray( v2.sum(0) ) [0] > 0
    i  = numpy.flatnonzero(numpy.logical_and(f1, f2))

    # Handle not having any features in common
    if i.sum() == 0: return numpy.zeros((v1.shape[0],v2.shape[0]), dtype='float')

    # Select only shared features from both matrices
    v1 = v1.transpose()[i].transpose().toarray()
    v2 = v2.transpose()[i].transpose().toarray()
    self.logger.debug('Reduced matrices from %d to %d features', orig_feats, v1.shape[1] )

    self.logger.debug('Calculating normals')
    n_v1 = [ (v, norm(v)) for v in v1 ]
    n_v2 = [ (v, norm(v)) for v in v2 ]

    self.logger.debug('Computing Distances')
    results = numpy.empty((v1.shape[0],v2.shape[0]), dtype='float')
    self.logger.debug('Output shape: %s', str(results.shape))
 
    def report(i, t): self.logger.debug('Processing entry %d', i+1 )

    for i,(p,np) in enumerate(timed_report(n_v1,10,report)):
      for j,(q,nq) in enumerate(n_v2):
        n = np * nq
        results[i,j] = numpy.dot(p,q) / n if n != 0 else 1.0

    self.logger.debug('Returning Results')
    return 1 - results

class dm_skew(distance_metric):
  # TODO: Implement dimensionality reduction as per cosine.
  __name__ = "skew"

  def __init__(self, alpha = 0.99):
    distance_metric.__init__(self)
    self.alpha = alpha

  def _params(self):
    p = distance_metric._params(self)
    p['alpha'] = self.alpha
    return p

  def vector_distances(self, v1, v2):
    self.logger.debug('Creating dense representation')
    orig_feats = v1.shape[1]
    # Determine shared features
    f1 = numpy.asarray( v1.sum(0) ) [0] > 0
    f2 = numpy.asarray( v2.sum(0) ) [0] > 0
    i  = numpy.flatnonzero(numpy.logical_and(f1, f2))

    # Select only shared features from both matrices
    v1 = v1.transpose()[i].transpose().toarray()
    v2 = v2.transpose()[i].transpose().toarray()
    self.logger.debug('Reduced matrices from %d to %d features', orig_feats, v1.shape[1] )

    self.logger.debug('Calculating distributions')
    s1 = [ float(v.sum()) for v in v1 ]
    s2 = [ float(v.sum()) for v in v2 ]
    # Replace an empty distribution with a uniform distribution.
    # Empty instances are a problem all the way back at the dataset level.
    n1 = [ self.alpha * ( (v/s) if s > 0 else numpy.ones_like(v) ) for (v,s) in izip(v1, s1) ]
    n2 = [ ( (v/s) if s > 0 else numpy.ones_like(v) ) for (v,s) in izip(v2,s2) ]
    n3 = [ (1-self.alpha) * q for q in n2 ]

    self.logger.debug('Computing Distances')
    results = numpy.empty((v1.shape[0],v2.shape[0]), dtype='float')

    def report(i, t): self.logger.debug('Processing entry %d', i+1 )

    for i,r_p in enumerate(timed_report(n1,10,report)):
      for j,(q, r_q) in enumerate(izip(n2, n3)):
        r = r_p + r_q
        r = r[q>0]
        q = q[q>0]
        results[i,j] = numpy.dot(q,numpy.log(q/r))

    self.logger.debug('Returning Results')
    return results

class dm_outofplace(distance_metric):
  """
  Ranklist-based distance metric described in Cavnar & Trenkel 1994.
  """
  __name__ = "OutOfPlace"
  # TODO: This could probably be further optimized for sparse data.
  #       In particular, we should be able to again reduce the set of features
  #       being considered. Features present in neither set will always rank tied last,
  #       so we should be able to separately calculate their contribution to the metric
  #       rather than naively compute it. Right now this metric is not really tractable
  #       for big,sparse spaces.
  
  @staticmethod
  def ranklist(v):
    v = v.toarray()[0]
    ordered_indices = reversed(numpy.argsort(v))
    output = numpy.empty(len(v),dtype=float)

    order = 1
    indices = [ ordered_indices.next() ]
    value = v[indices[0]] 

    for index in ordered_indices: 
      if v[index] == value:
        indices.append(index)
      else:
        score_sum = sum(range(len(indices)))
        score = order + (float(score_sum) / len(indices))
        for i in indices:
          output[i] = score

        order = order + len(indices)
        indices = [ index ]
        value = v[ index ]

    # finish last sequence
    score_sum = sum(range(len(indices)))
    score = order + (float(score_sum) / len(indices))
    for i in indices:
      output[i] = score

    return output
      
  def basic(self, p, q):
    p_rl = dm_outofplace.ranklist(p)
    q_rl = dm_outofplace.ranklist(q)
    result = self.fast(p_rl, q_rl) 
    return result

  def fast(self, p_rl, q_rl):
    return numpy.sum(numpy.abs(p_rl - q_rl))

  def vector_distances(self, v1, v2):
    rl_v1 = [ dm_outofplace.ranklist(v) for v in v1 ]
    rl_v2 = [ dm_outofplace.ranklist(v) for v in v2 ]

    results = numpy.empty((v1.shape[0],v2.shape[0]))
    for i,(q_rl) in enumerate(rl_v1):
      for j,(r_rl) in enumerate(rl_v2):
        results[i,j] = self.fast(q_rl, r_rl)
    return results

# TODO: Tau has been dead code for some time now. Would be nice to 
#       bring it back to a workable state.
class dm_tau(distance_metric):
  __name__ = "Kendall's Tau"
  def basic(self, p, q):
    assert len(p) == len(q)
    count = 0
    n = len(p)
    for i in xrange(n):
      for j in xrange(i,n):
        if (p[i] < p[j] and q[i] > q[j]) or (p[i] > p[j] and q[i] < q[j]):
          count += 1
    return count / (n*(n-1)/2)

  def fast(self, q_prime, r_prime):
    concordances = numpy.logical_xor(q_prime,r_prime)
    tau = numpy.sum(concordances) / len(q_prime)
    return tau

  def _pairwise_dir(self, x):
    chunks = []
    for i in xrange(1,len(x)):
      chunks.append(x[:i] > x[-i:])
    return concatenate(chunks)

  def vector_distances(self, v1, v2):
    v1_prime = [ self._pairwise_dir(v) for v in v1 ]
    v2_prime = [ self._pairwise_dir(v) for v in v2 ]
    results = numpy.empty((len(v1),len(v2)))
    for i,q_prime in enumerate(v1_prime):
      for j,r_prime in enumerate(v2_prime):
        results[i,j] = self.fast(q_prime, r_prime)
    return results

###########
# Distance metrics from rpy
###########
"""
try:
  import rpy
  class dm_r(distance_metric):
    __name__="R cor"
    def __init__(self, method):
      self.method = method

    def basic(self, p, q):
      return rpy.r.cor(p,q,method = self.method)

    def vector_distances(self, v1, v2):
      results = numpy.empty((len(v1),len(v2)))
      for i,q in enumerate(v1):
        for j,r in enumerate(v2):
          results[i,j] = rpy.r.cor(q,r,method = self.method) 
      return results
      
  class dm_r_tau(dm_r):
    __name__="Kendall's Tau (R)"
    def __init__(self):
      dm_r.__init__(self, "kendall")
  __all__.append('dm_r_tau')
except ImportError:
  pass
"""

###########
# Distance metrics from biopython 
###########
try:
  import Bio.Cluster as bc
  class dm_biopython(distance_metric):
    __name__="BioPython distancematrix"
    def __init__(self, dist):
      self.dist = dist

    def basic(self, p, q):
      return bc.distancematrix((p,q), dist = self.dist)

    def vector_distances(self, v1, v2):
      if v1.dtype == numpy.uint64 or v2.dtype == numpy.uint64:
        raise TypeError, "Biopython cannot deal with uint64 arrays"
      results = numpy.empty((len(v1),len(v2)))
      for i,q in enumerate(v1):
        for j,r in enumerate(v2):
          results[i,j] = bc.distancematrix((q,r), dist=self.dist)[1][0]
      return results

  class dm_bp_tau(dm_biopython):
    """ Kendall's Tau tb (standard tie correction included), as implemented
    in BioPython"""
    __name__="Kendall's Tau (BioPython)"
    def __init__(self):
      dm_biopython.__init__(self, "k")
  __all__.append('dm_bp_tau')

  class dm_bp_rho(dm_biopython):
    """ Spearman's Rho rsb (tie correction included), as implemented
    in BioPython"""
    __name__="Spearman's Rho (BioPython)"
    def __init__(self):
      dm_biopython.__init__(self, "s")
  __all__.append('dm_bp_rho')
except ImportError:
  pass


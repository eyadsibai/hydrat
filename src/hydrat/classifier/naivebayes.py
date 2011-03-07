"""
Mutlinomial Naive Bayes
see McCallum and Nigam (1998) "A Comparison of Event Models for Naive Bayes Text Classification"

..todo: Implement other event models

Marco Lui Feb 2011
"""
import numpy as np
import scipy.sparse as sp

from hydrat.classifier.abstract import Learner, Classifier

def logfac(a):
  return np.sum(np.log(np.arange(1,a+1)))
logfac = np.frompyfunc(logfac, 1, 1)

class NBEventModel(object):
  """
  Abstract algorithm class for Naive Bayes event models.
  Implements two methods, for calculaing class priors and 
  term priors on the basis of the feature and class matrixes.
  Each method should return the log(P(C)) and log(P(t|C)) 
  respectively.
  Assumes that both the feature and the class matrix are
  scipy.sparse.csr_matrix instances.
  """
  def __init__(self):
    if not hasattr(self, '__name__'):
      self.__name__ = self.__class__.__name__

  def class_priors(self, fm, cm):
    raise NotImplementedError

  def term_priors(self, fm, cm):
    raise NotImplementedError

class Multinomial(NBEventModel):
  def class_priors(self, fm, cm):
    return np.log(cm.sum(0))

  def term_priors(self, fm, cm):
    v = fm.shape[1]
    prod = (fm.T * cm).todense()
    ptc = np.log(1 + prod) - np.log(v + prod.sum(0))
    return ptc
 
class UniformClassPrior(NBEventModel):
  def class_priors(self, fm, cm):
    return np.log(np.ones((cm.shape[1],),dtype=cm.dtype))

class MultinomialBoolean(Multinomial):
  """
  Convert term matrix to boolean before applying multinomial model.
  ..todo: references
  """
  def term_priors(self, fm, cm):
    fm_ = fm.copy()
    fm_.data = (fm_.data > 0).astype('int64')
    return Multinomial.term_priors(self, fm_, cm)

class NaiveBayesL(Learner):
  __name__ = "naivebayes" 
  def __init__(self, eventmodel):
    Learner.__init__(self)
    self.eventmodel = eventmodel

  def _params(self):
    return {'eventmodel':self.eventmodel.__name__}

  def _learn(self, fm, cm):
    # TODO: Do a collapse of unused classes
    tot_cl = cm.shape[1]
    used_cl = np.flatnonzero(cm.sum(0) > 0)
    cm = sp.csr_matrix(cm[:,used_cl], dtype='int32')
    pc = self.eventmodel.class_priors(fm, cm)
    ptc = self.eventmodel.term_priors(fm, cm)
    return NaiveBayesC(pc, ptc, tot_cl, used_cl)

  def _check_installed(self):
    pass

class NaiveBayesC(Classifier):
  __name__ = "naivebayes" 
  def __init__(self, pc, ptc, tot_cl, used_cl):
    """
    pc: class priors
    ptc: term priors
    tot_cl: total number of classes in the larger class space
    used_cl: indexes of classes used in the larger class space
    """
    Classifier.__init__(self)
    self.pc = pc
    self.ptc = ptc
    self.tot_cl = tot_cl
    self.used_cl = used_cl

  def _classify(self, fm):
    lf = fm.copy()
    lf.data = logfac(lf.data).astype(float)
    pdc = fm * self.ptc - lf.sum(1)
    cl = np.argmax(pdc + self.pc, axis=1)
    cm = np.zeros((fm.shape[0], self.tot_cl), dtype=bool)
    cm[np.arange(cm.shape[0]), self.used_cl[np.array(cl)[:,0]]] = True
    return cm

def multinomialL(): return NaiveBayesL(Multinomial())
def multinomialBL(): return NaiveBayesL(MultinomialBoolean())


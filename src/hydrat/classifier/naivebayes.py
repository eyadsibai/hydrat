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

class NaiveBayesL(Learner):
  __name__ = "naivebayes" 

  def _learn(self, fm, cm):
    v = fm.shape[1]
    cm = sp.csr_matrix(cm, dtype='int32')
    prod = (fm.T * cm).todense()
    p_t_c = np.log(1 + prod) - np.log(v + prod.sum(0))
    # TODO: Replace this smoothing with a collapse of unused classes. Will be faster!
    p_c = np.log(cm.sum(0) + 1)
    return NaiveBayesC(p_c, p_t_c)

  def _params(self):
    return {}

  def _check_installed(self):
    pass

class NaiveBayesC(Classifier):
  __name__ = "naivebayes" 
  def __init__(self, pc, ptc):
    Classifier.__init__(self)
    self.pc = pc
    self.ptc = ptc

  def _classify(self, fm):
    lf = fm.copy()
    lf.data = logfac(lf.data).astype(float)
    pdc = fm * self.ptc - lf.sum(1)
    cl = np.argmax(pdc + self.pc, axis=1)
    cm = np.zeros((fm.shape[0], self.pc.shape[1]), dtype=bool)
    cm[np.arange(cm.shape[0]), np.array(cl)[:,0]] = True
    return cm


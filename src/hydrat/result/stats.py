import numpy as np
from scipy.stats import chi2
def mcnemar_test(ec_a, ec_b):
  """
  McNemar's test. Accepts multidimensional arrays as well, so we can compute the p-value
  for multiple classes at once.

  see
  Detterich, Thomas G, Statistical Tests for Comparing Supervised Classification Learning Algorithms. ( 1997).

  NOTE: Classes with no instances return a p-value of 0
  """
  nc_a = np.logical_not(ec_a)
  nc_b = np.logical_not(ec_b)

  n00 = np.logical_and(nc_a, nc_b).sum(axis=0)
  n01 = np.logical_and(nc_a, ec_b).sum(axis=0)
  n10 = np.logical_and(ec_a, nc_b).sum(axis=0)
  n11 = np.logical_and(ec_a, ec_b).sum(axis=0)

  # Ignore division by zero on the computation of the statistic, it is the result of unused classes
  prev_set = np.seterr(divide='ignore')
  stat = np.square(np.abs(n01 - n10) - 1 ) / np.array(n01+n10, dtype=float)
  np.seterr(**prev_set)

  rv = chi2(1)
  p = rv.sf(stat)
  return p

def mcnemar(interpreter, tsr_a, tsr_b, perclass = False):
  if perclass:
    correct_a = np.nansum(tsr_a.overall_correct(interpreter),axis=2)
    correct_b = np.nansum(tsr_b.overall_correct(interpreter),axis=2)
    return mcnemar_test(correct_a, correct_b)
  else:
    ec_a = np.nansum(tsr_a.overall_correct(interpreter),axis=2).all(axis=1).astype(bool)
    ec_b = np.nansum(tsr_b.overall_correct(interpreter),axis=2).all(axis=1).astype(bool)
    return mcnemar_test(ec_a, ec_b)

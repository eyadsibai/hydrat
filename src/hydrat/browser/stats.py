import numpy as np
from scipy.stats import chi2

def mcnemar(interpreter, tsr_a, tsr_b):
  """
  McNemar's test

  see
  Detterich, Thomas G, Statistical Tests for Comparing Supervised Classification Learning Algorithms. ( 1997).
  """
  ec_a = tsr_a.overall_exactly_correct(interpreter).sum(axis=1).astype(bool)
  ec_b = tsr_b.overall_exactly_correct(interpreter).sum(axis=1).astype(bool)
  nc_a = np.logical_not(ec_a)
  nc_b = np.logical_not(ec_b)

  n00 = np.logical_and(nc_a, nc_b).sum()
  n01 = np.logical_and(nc_a, ec_b).sum()
  n10 = np.logical_and(ec_a, nc_b).sum()
  n11 = np.logical_and(ec_a, ec_b).sum()

  stat = np.square(np.abs(n01 - n10) - 1 ) / float(n01+n10)
  rv = chi2(1)
  return rv.sf(stat)

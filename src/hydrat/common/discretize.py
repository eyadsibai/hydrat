"""
Discretization functions
"""

import numpy

def bernoulli(v):
  nonzero = v.nonzero()[0]
  zero = numpy.array(list(set(range(v.shape[0])) - set(nonzero)))
  return [zero, nonzero]

class UniformBand(object):
  def __init__(self, bands):
    self.__name__ = 'uniform%dband' % bands
    self.bands = bands

  def __call__(self, v):
    limit = float(numpy.max(v.data) + 1)
    bins = numpy.digitize(v, numpy.arange(0, limit, limit/self.bands))
    r = numpy.empty((self.bands, len(v)), dtype=bool)
    for i in range(self.bands):
      r[i] = (bins == (i+1))
    return r

class EquisizeBand(object):
  def __init__(self, bands):
    self.__name__ = 'equisize%dband' % bands
    self.bands = bands

  def __call__(self, v):
    r = numpy.empty((self.bands, v.shape[0]), dtype=bool)
    band_size = 100.0 / (self.bands)
    for i in range(self.bands):
      r[i] = numpy.logical_and( (i * band_size) <= v, v < (i * (band_size + 1)) )
    return r

import numpy
from time import time

def proportion_allocate(itemlist, proportions, randomise = False, randomseed = None):
  try: 
    denominator = float(sum(proportions))
  except TypeError:
    raise TypeError, "Proportions not number values"
  totalitems = len(itemlist)
  partitionsizes = map(lambda x: int(x * totalitems // denominator) , proportions)
  partitionsizes[0] = totalitems - sum(partitionsizes[1:]) #Account for rounding errors by favoring first partition
  partitions = []
  if randomise:
    import random
    if randomseed != None:
      random.seed(randomseed)
    itemlist = itemlist[:]
    random.shuffle(itemlist)
  for i in xrange(len(partitionsizes)):
    partition_items = itemlist[:partitionsizes[i]]
    itemlist = itemlist[partitionsizes[i]:]
    partitions.append(partition_items)
  assert len(itemlist) == 0, "Elements left in itemlist: " + str(len(itemlist))
  return partitions

def progress(seq, count, callback):
  assert callable(callback)
  assert isinstance(count, int)
  call_indices = []
  step = len(seq) / float(count)
  curr = 0 
  while curr < len(seq):
    call_indices.append(int(curr))
    curr += step
  for i,x in enumerate(seq):
    if i in call_indices: callback(i, len(seq))
    yield x

def timed_report(seq, interval, callback):
  """
  Call a callback if a certain amount of time has elapsed in processing a sequence
  """
  assert callable(callback)
  try:
    total = len(seq)
  except TypeError:
    total = None
  next_call = time() + interval

  for i,x in enumerate(seq):
    t = time()
    if t > next_call:
      callback(i, total)
      next_call = t + interval 
    yield x
    
def entropy(v):
  assert isinstance(v, numpy.ndarray)
  s = v.sum()
  if s == 0: return 0.0
  p = v / float(s)
  o = numpy.seterr(divide='ignore', invalid='ignore')
  r = -1 * numpy.nansum( p * numpy.log2(p))
  numpy.seterr(**o)
  return r



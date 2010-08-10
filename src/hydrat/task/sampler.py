import random
import time
import numpy
from sys import maxint

def isOneofM(class_map):
  """
  Check that a class_map represents a 1 of M class mapping
  @type class_map: boolean class map
  @rtype: boolean
  """
  assert(class_map.dtype == 'bool')
  class_counts = class_map.sum(1)
  return (class_counts == 1).all()

def stratify(class_map):
  """
  Takes a classmap where documents may belong to multiple classes
  and defines a new classmap where each document belongs to a single
  class, which represents the set of original classes the document 
  belonged to.
  @type class_map: boolean class_map
  @rtype: boolean class_map
  """
  assert(class_map.dtype == 'bool')
  num_docs = class_map.shape[0]

  strata_identifiers = numpy.zeros(num_docs)
  for i, row in enumerate(class_map):
    identifier = 0
    for entry in row:
      identifier << 1
      if entry:
        identifier += 1
    strata_identifiers[i] = identifier
  assert len(row) == class_map.shape[1]

  unique_identifiers  = set(strata_identifiers)
  num_stratas         = len(unique_identifiers)
  strata_indices      = dict( (strata, i) for i,strata in enumerate(unique_identifiers))

  strata_map          = numpy.zeros((num_docs,num_stratas), dtype = 'bool')
  for doc_index, id in enumerate(strata_identifiers):
    strata_index                         = strata_indices[id]
    strata_map[doc_index, strata_index]  = True
  return strata_map

def partition(num_items, weights, probabilistic = False, rng = None):
  """
  Partition a number of items into partitions according to weights
  @return: map of doc_index -> partition membership
  @rtype: numpy boolean array (items * partitions) 
  """
  probabilities = weights / float(weights.sum())
  if rng is None:
    rng = random.Random()
  # Changed from -maxint-1 to maxint because newer numpy requires positive seed.
  numpy.random.seed(rng.randint(0, maxint))
  num_parts   = len(probabilities)
  partition_map = numpy.zeros((num_items,num_parts), dtype = 'bool')
  if probabilistic:
    c = numpy.cumsum(probabilities)
    r = numpy.random.random(num_items)
    partition_indices = numpy.searchsorted(c, r)
    for doc_index, part_index in enumerate(partition_indices):
      partition_map[doc_index, part_index] = True
  else:
    partition_sizes = numpy.floor(probabilities * num_items).astype(int)
    items_partitioned = partition_sizes.sum()
    gap = num_items - items_partitioned
    # Distribute the gap amongst the biggest partitions, adding one to each
    # This behaviour was deprecated as it does not behave well when partitions
    # all have the same size. The later partitions get favored by nature of the
    # argsort.
    # distribute_gap_to = numpy.argsort(partition_sizes) >= (num_parts - gap)
    # Randomly distribute it 
    distribute_gap_to = numpy.concatenate(( numpy.ones(gap, dtype=bool), numpy.zeros(num_parts-gap, dtype=bool) )) 
    numpy.random.shuffle(distribute_gap_to)
    partition_sizes[distribute_gap_to] += 1
    assert partition_sizes.sum() == num_items
    indices = numpy.arange(num_items) 
    numpy.random.shuffle(indices)
    index        = 0
    for part_index, part_size in enumerate(partition_sizes):
      for i in xrange(part_size):
        doc_index = indices[index]
        partition_map[doc_index, part_index] = True
        index += 1
  return partition_map

def allocate(strata_map, weights, probabilistic = False, rng = None):
  """
  Stratified allocation of items into partitions
  @return: map of doc_index -> partition membership
  @rtype: numpy boolean array
  """
  assert isOneofM(strata_map)
  num_parts             = len(weights)
  num_docs, num_strata  = strata_map.shape
  part_map              = numpy.empty((num_docs,num_parts),dtype='bool')

  for strata_index in xrange(num_strata):
    strata_items        = strata_map[:,strata_index]
    strata_num_items    = strata_items.sum() 
    strata_doc_indices  = numpy.arange(len(strata_items))[strata_items]
    strata_part_map     = partition(strata_num_items, weights, probabilistic, rng)
    for inner_index, doc_index in enumerate(strata_doc_indices):
      part_map[doc_index] = strata_part_map[inner_index]
  return part_map

from hydrat.task.taskset import TaskSet
from hydrat.task.task import InMemoryTask


class Partitioning(object):
  """ Represents a partitioning on a set of data """
  def __init__(self, class_map, partitions, metadata):
    self.class_map = class_map
    self.parts = partitions
    self.metadata = dict(class_map.metadata)
    self.metadata.update(metadata)

  def __call__(self, feature_map, additional_metadata={}):
    """Transform a feature map into a TaskSet"""
    # Check the number of instances match
    assert feature_map.raw.shape[0] == self.parts.shape[0]
    # Check the feature map and class map are over the same dataset
    assert feature_map.metadata['dataset_uuid'] == self.class_map.metadata['dataset_uuid']
    tasklist = []
    metadata = dict(self.metadata)
    metadata.update(feature_map.metadata)
    metadata.update(additional_metadata)
    for i in range(self.parts.shape[1]):
      partition  = self.parts[:,i]
      test_ids   = partition 
      train_ids  = numpy.logical_not(partition)
      md = dict(metadata)
      md['part_index'] = i
      tasklist.append( InMemoryTask   ( feature_map.raw
                                      , self.class_map.raw
                                      , train_ids
                                      , test_ids 
                                      , md 
                                      )
                     )
    return TaskSet(tasklist, metadata)

class Sampler(object):
  """ A sampler takes a class map and produces partitions 
      TODO: Does sampling have to be defined in terms of a class map?
  """
  def __init__(self, seed=None):
    if seed is None: seed = time.time()
    rng   = random.Random()
    rng.seed(seed)
    self.rng = rng
    self.seed = seed

  def __call__(self, class_map):
    return self.sample(class_map)
    
  def sample(self, class_map):
    """ Classes deriving sampler must implement this """
    raise NotImplementedError

class TrainTest(Sampler):
  def __init__(self, ratio=4, seed=None):
    Sampler.__init__(self, seed)
    self.ratio = ratio

  def sample(self, class_map):
    strata_map = stratify(class_map.raw)
    partition_proportions = numpy.array([ratio, 1])
    parts  = allocate( strata_map
                     , partition_proportions
                     , probabilistic = False
                     , rng=self.rng
                     ) 
    metadata = dict()
    metadata['task_type'] = 'train_test'
    metadata['seed'] = self.seed
    return Partitioning(class_map, parts, metadata) 

class CrossValidate(Sampler):
  def __init__(self, folds=10, seed=None):
    Sampler.__init__(self, seed)
    self.folds = folds

  def sample(self, class_map):
    strata_map = stratify(class_map.raw)
    partition_proportions = numpy.array([1] * self.folds )
    folds = allocate( strata_map
                    , partition_proportions
                    , probabilistic = False
                    , rng=self.rng
                    ) 
    metadata = dict()
    metadata['task_type'] = 'crossvalidate'
    metadata['seed'] = self.seed
    return Partitioning(class_map, folds, metadata) 

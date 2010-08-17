import time
import numpy

from hydrat.common.sampling import partition

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

  # Build strata identifiers, which are a bit patern indicating 
  # set-of-classes membership
  strata_identifiers = numpy.zeros(num_docs)
  for i, row in enumerate(class_map):
    identifier = 0
    for entry in row:
      identifier = identifier << 1
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

def allocate(strata_map, weights, probabilistic = False, rng = None):
  """
  Allocation of items into partitions
  Stratification must be performed separately, but this function will
  only work on 1-of-m classmaps
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
    strata_part_map     = partition(strata_num_items, weights, probabilistic, rng=rng)
    for inner_index, doc_index in enumerate(strata_doc_indices):
      part_map[doc_index] = strata_part_map[inner_index]
  return part_map

from hydrat.task.taskset import TaskSet
from hydrat.task.task import InMemoryTask

class Partitioning(object):
  """ Represents a partitioning on a set of data """
  def __init__(self, class_map, partitions, metadata):
    # partitions is a 3-d array. instances X partitions X train/test(note order!)
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
      train_ids  = self.parts[:,i,0]
      test_ids   = self.parts[:,i,1]

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
  def __init__(self, rng):
    self.rng = rng

  def __call__(self, class_map):
    return self.sample(class_map)
    
  def sample(self, class_map):
    """ Classes deriving sampler must implement this """
    raise NotImplementedError

def membership_vector(superset, subset):
  return numpy.fromiter((s in subset for s in superset), dtype=bool)
  
class PresetSplit(Sampler):
  def __init__(self, split, rng=None):
    Sampler.__init__(self, rng)
    self.split = split

  def sample(self, class_map):
    metadata = dict()
    metadata['task_type'] = 'preset_split'
    metadata['rng_state'] = self.rng.get_state()
    return Partitioning(class_map, self.split, metadata) 

class TrainTest(Sampler):
  def __init__(self, ratio=4, rng=None):
    Sampler.__init__(self, rng)
    self.ratio = ratio

  def sample(self, class_map):
    strata_map = stratify(class_map.raw)
    partition_proportions = numpy.array([self.ratio, 1])
    parts  = allocate( strata_map
                     , partition_proportions
                     , probabilistic = False
                     , rng=self.rng
                     ) 
    # Build train/test by stacking to the correct shape
    parts = numpy.dstack((parts[:,0], parts[:,1])).swapaxes(0,1)
    metadata = dict()
    metadata['task_type'] = 'train_test'
    metadata['rng_state'] = self.rng.get_state()
    return Partitioning(class_map, parts, metadata) 

class CrossValidate(Sampler):
  def __init__(self, folds=10, rng=None):
    Sampler.__init__(self, rng)
    self.folds = folds

  def sample(self, class_map):
    strata_map = stratify(class_map.raw)
    partition_proportions = numpy.array([1] * self.folds )
    folds = allocate( strata_map
                    , partition_proportions
                    , probabilistic = False
                    , rng=self.rng
                    ) 
    # We build train/test. Axis 0 is train, so it is the logical not of each fold, which represents
    # the test instances.
    folds = numpy.dstack((numpy.logical_not(folds), folds))
    metadata = dict()
    metadata['task_type'] = 'crossvalidate'
    metadata['rng_state'] = self.rng.get_state()
    return Partitioning(class_map, folds, metadata) 

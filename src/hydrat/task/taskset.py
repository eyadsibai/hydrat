class TaskSet(object):
  """
  Collates task objects and their metadata
  """
  def __init__( self
              , tasks 
              , metadata
              ):
    self.tasks = tasks
    self.metadata = dict(metadata)

  def __eq__(self, other):
    raise NotImplementedError

def from_partitions( partitions
                   , feature_map
                   , class_map
                   , sequence = None
                   , metadata = {}
                   ):
  #TODO: Do something with the sequence!!!
  # partitions is a 3-d array. instances X partitions X train/test(note order!)
  # Check the number of instances match
  assert feature_map.raw.shape[0] == self.parts.shape[0]
  # Check the feature map and class map are over the same dataset
  assert feature_map.metadata['dataset'] == self.class_map.metadata['dataset']

  md = dict(class_map.metadata)
  md.update(feature_map.metadata)
  md.update(metadata)

  tasklist = []
  for i in range(self.parts.shape[1]):
    train_ids  = self.parts[:,i,0]
    test_ids   = self.parts[:,i,1]

    pmd = dict(metadata)
    pmd['part_index'] = i
    tasklist.append( InMemoryTask   ( feature_map.raw
                                    , self.class_map.raw
                                    , train_ids
                                    , test_ids 
                                    , pmd 
                                    )
                    )
  return TaskSet(tasklist, md)

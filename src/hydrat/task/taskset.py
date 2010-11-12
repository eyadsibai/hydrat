from task import InMemoryTask
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

# TODO: Split the metadata management out from the 
#       core problem of producing new tasks.
def from_partitions( partitions
                   , feature_map
                   , class_map
                   , sequence = None
                   , metadata = {}
                   ):
  # partitions is a 3-d array. instances X partitions X train/test(note order!)
  # Check the number of instances match
  assert feature_map.raw.shape[0] == partitions.shape[0]
  # Check the feature map and class map are over the same dataset
  # assert feature_map.metadata['dataset'] == class_map.metadata['dataset']

  md = dict(class_map.metadata)
  md.update(feature_map.metadata)
  md.update(metadata)

  tasklist = []
  for i in range(partitions.shape[1]):
    train_ids  = partitions[:,i,0]
    test_ids   = partitions[:,i,1]

    pmd = dict(metadata)
    # TODO: is this really where we should be labelling the tasks?
    # NOTE: This index is currently being used by Store._get_Task. Dirty.
    pmd['index'] = i
    tasklist.append( InMemoryTask   ( feature_map.raw
                                    , class_map.raw
                                    , train_ids
                                    , test_ids 
                                    , pmd 
                                    , sequence = sequence
                                    )
                    )
  return TaskSet(tasklist, md)

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



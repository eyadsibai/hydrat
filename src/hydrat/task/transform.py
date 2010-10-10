from hydrat.task.taskset import TaskSet
from hydrat.task.task import Task

def transform_task(task, transformer):
  transformer.learn(task.train_vectors, task.train_classes)
  t = Task()
  affected = ['train_vectors', 'test_vectors']
  for slot in Task.__slots__:
    if slot in affected:
      setattr(t, slot, transformer.apply(getattr(task, slot)))
    else:
      setattr(t, slot, getattr(task, slot))
  # Separately update metadata
  t.metadata      = dict(task.metadata)
  t.metadata['feature_desc']+=(transformer.__name__,)
  return t
  
def transform_taskset(taskset, transformer):
  metadata = update_metadata(taskset.metadata, transformer)
  tasklist = [ transform_task(t, transformer) for t in taskset.tasks ]
  return TaskSet(tasklist, metadata)

def update_metadata(metadata, transformer):
  metadata = dict(metadata)
  # Eliminate feature name
  if 'feature_name' in metadata: 
    raise ValueError, "Should not be encountering feature_name"
  metadata['feature_desc']+=(transformer.__name__,)
  return metadata




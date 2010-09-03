from hydrat.task.taskset import TaskSet
from hydrat.task.task import Task

def transform_task(task, transformer):
  transformer.learn(task.train_vectors, task.train_classes)
  t = Task()
  t.train_vectors = transformer.apply(task.train_vectors)
  t.test_vectors  = transformer.apply(task.test_vectors)
  t.train_classes = task.train_classes
  t.test_classes  = task.test_classes
  t.train_indices = task.train_indices
  t.test_indices  = task.test_indices
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




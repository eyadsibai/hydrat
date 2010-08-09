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
  # Eliminate feature name
  if 'feature_name' in t.metadata: 
    t.metadata['feature_desc'] = ( t.metadata['feature_name'], )
    del t.metadata['feature_name']
  t.metadata['feature_desc']+=(transformer.__name__,)
  return t
  
def transform_taskset(taskset, transformer):
  tasklist = [ transform_task(t, transformer) for t in taskset.tasks ]
  metadata = dict(taskset.metadata)
  # Eliminate feature name
  if 'feature_name' in metadata: 
    if 'feature_desc' in metadata:
      raise ValueError, "Metadata contained both feature_name and feature_desc"
    metadata['feature_desc'] = ( metadata['feature_name'], )
    del metadata['feature_name']
  metadata['feature_desc']+=(transformer.__name__,)
  return TaskSet(tasklist, metadata)


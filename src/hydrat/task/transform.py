from hydrat.task.taskset import TaskSet
from hydrat.task.task import Task
import time

def transform_task(task, transformer):
  start = time.time()
  transformer.learn(task.train_vectors, task.train_classes)
  learn_time = time.time() - start

  t = Task()
  affected = ['train_vectors', 'test_vectors']
  start = time.time()
  for slot in Task.__slots__:
    if slot in affected:
      setattr(t, slot, transformer.apply(getattr(task, slot)))
    else:
      setattr(t, slot, getattr(task, slot))
  apply_time = time.time() - start

  # Separately update metadata
  t.metadata      = dict(task.metadata)
  t.metadata['feature_desc']+=(transformer.__name__,)
  if 'transform_learn_time' not in t.metadata:
    t.metadata['transform_learn_time'] = {}
  if 'transform_apply_time' not in t.metadata:
    t.metadata['transform_apply_time'] = {}
  t.metadata['transform_learn_time'][transformer.__name__] = learn_time
  t.metadata['transform_apply_time'][transformer.__name__] = apply_time
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




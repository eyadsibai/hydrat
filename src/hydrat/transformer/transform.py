from hydrat.datamodel import TaskSet, Task

class Transform(TaskSet):
  def __init__(self, taskset, transformer):
    self.taskset = taskset 
    self.transformer = transformer

  @property
  def metadata(self):
    # TODO: how about parametrized transformers?
    metadata = dict(taskset.metadata)
    metadata['feature_desc'] += (transformer.__name__,)
    return metadata
    
  @property
  def tasks(self):
    # TODO: Work out how to get the required weights, and how to extend them back
    # TODO: Work out why we needed add_args, and what to do with it now
    # TODO: Timing of the component parts
    tasks = []
    for task in self.taskset.tasks:
      # Transform each task
      add_args = {}
      add_args['sequence'] = task.train_sequence
      add_args['indices'] = task.train_indices

      # Patch transformer with our known weights
      weights = self.transformer.weights
      self.transformer.weights = task.weights

      self.transformer._learn(task.train_vectors, task.train_classes, add_args)

      t = Task()
      for slot in Task.__slots__:
        if slot.endswith('vectors'):
          if slot.startswith('train'):
            add_args['sequence'] = task.train_sequence
            add_args['indices'] = task.train_indices
          elif slot.startswith('test'):
            add_args['sequence'] = task.test_sequence
            add_args['indices'] = task.test_indices
          setattr(t, slot, self.transformer._apply(getattr(task, slot), add_args))
        else:
          setattr(t, slot, getattr(task, slot))
      t.metadata      = dict(task.metadata)

      # Copy weights back into task
      self.transformer.weights = weights
      tasks.append(t)
    return tasks

from hydrat.datamodel import TaskSet, BasicTask

class Transform(TaskSet):
  def __init__(self, taskset, transformer):
    self.taskset = taskset 
    self.transformer = transformer

  """
  # TODO: figure out why this is here in the first place.
  def __getattr__(self, key):
    if key in self.__dict__:
      return self.__dict__[key]
    else:
      return getattr(self.taskset, key)
  """

  @property
  def metadata(self):
    # TODO: how about parametrized transformers?
    metadata = dict(self.taskset.metadata)
    metadata['feature_desc'] += (self.transformer.__name__,)
    return metadata

  def __len__(self):
    return len(self.taskset)

  def __getitem__(self, key):
    # TODO: Work out why we needed add_args, and what to do with it now
    # TODO: Timing of the component parts
    task = self.taskset[key]
    # Transform each task
    add_args = {}
    add_args['sequence'] = task.train_sequence
    add_args['indices'] = task.train_indices

    # Patch transformer with our known weights
    # TODO: Why can't weights just be an argument?
    weights = self.transformer.weights
    self.transformer.weights = task.weights

    self.transformer._learn(task.train_vectors, task.train_classes, add_args)

    # Transform train vectors
    add_args = {}
    add_args['sequence'] = task.train_sequence
    add_args['indices'] = task.train_indices
    train_vectors = self.transformer._apply(task.train_vectors, add_args)

    # Transform test vectors
    add_args = {}
    add_args['sequence'] = task.test_sequence
    add_args['indices'] = task.test_indices
    test_vectors = self.transformer._apply(task.test_vectors, add_args)


    t = BasicTask(
      train_vectors, 
      task.train_classes, 
      task.train_indices,
      test_vectors, 
      task.test_classes, 
      task.test_indices,
      task.train_sequence, 
      task.test_sequence,
      metadata= dict(task.metadata)
      )

    # Copy weights back into task
    self.transformer.weights = weights
    return t

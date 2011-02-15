# experiment.py
# Marco Lui Feb 2011
#
# This class represents an experiment. Its intention is to act as the interface between the user
# and the udnerlying store at the level of managing tasks and results. The dataset abstraction is
# delegated to the DataProxy object.

from hydrat.datamodel import TaskSetResult, Result
from hydrat.common.pb import ProgressIter

# TODO: Refactor in a way that allows access to per-fold classifiers
class Experiment(TaskSetResult):
  def __init__(self, taskset, learner=None):
    self.taskset = taskset
    self.learner = learner
    self._results = None

  @property
  def metadata(self):
    """ Result object metadata """
    result_metadata = dict(self.taskset.metadata)
    result_metadata['learner'] = self.learner.__name__
    result_metadata['learner_params'] = self.learner.params
    return result_metadata 
    
  @property
  def results(self):
    if self._results is None:
      self.run()
    return self._results

  def run(self, add_args = None):
    label = "Experiment: %s %s" % (self.learner.__name__, self.learner.params)
    results = []
    for task in ProgressIter(self.taskset.tasks, label):
      if add_args is None:
        add_args = {}

      # TODO: Decide if it ever makes sense to allow add_args that don't have to be
      # normalized by the split.
      # NOTE: Sequence requires special treatment because it has to be normalized in both
      # axes.
      train_add_args = dict( (k, v[task.train_indices]) for k,v in add_args.items())
      classifier = self.learner( task.train_vectors, task.train_classes,\
          sequence=task.train_sequence, indices=task.train_indices, **train_add_args)

      test_add_args = dict( (k, v[task.test_indices]) for k,v in add_args.items())
      classifications = classifier( task.test_vectors,\
          sequence=task.test_sequence, indices=task.test_indices, **test_add_args)

      results.append(Result.from_task(task, classifications, dict(classifier.metadata)))
          
    self._results = results
    return results


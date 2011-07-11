# experiment.py
# Marco Lui Feb 2011
#
# This class represents an experiment. Its intention is to act as the interface between the user
# and the udnerlying store at the level of managing tasks and results. The dataset abstraction is
# delegated to the DataProxy object.

from hydrat.datamodel import TaskSetResult, Result
from hydrat.common.pb import ProgressIter

class ExperimentFold(object):
  def __init__(self, task, learner, add_args={}):
    self.task = task
    self.learner = learner
    self.add_args = add_args

  @property
  def classifier(self):
    train_add_args = dict( (k, v[self.task.train_indices]) for k,v in self.add_args.items())
    classifier = self.learner( self.task.train_vectors, self.task.train_classes,\
        sequence=self.task.train_sequence, indices=self.task.train_indices, **train_add_args)
    return classifier

  @property
  def result(self):
    classifier = self.classifier
    test_add_args = dict( (k, v[self.task.test_indices]) for k,v in self.add_args.items())
    classifications = classifier( self.task.test_vectors,\
        sequence=self.task.test_sequence, indices=self.task.test_indices, **test_add_args)

    return Result.from_task(self.task, classifications, dict(classifier.metadata))

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

  @property
  def folds(self):
    folds = []
    for task in self.taskset.tasks:
      folds.append(ExperimentFold(task, self.learner))
    return folds

  def run(self, add_args = None):
    results = []
    print "Experiment: %s %s" % (self.learner.__name__, self.learner.params)
    for fold in ProgressIter(self.folds, ''):
      results.append(fold.result)
    self._results = results
    return results


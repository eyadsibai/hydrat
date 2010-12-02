import time
import logging
import uuid
import numpy
from hydrat.result.result import result_from_task 
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.common.pb import ProgressIter
logger = logging.getLogger(__name__)

def run_task( learner, task, add_args=None):
  if add_args is None:
    add_args = {}

  # TODO: Decide if it ever makes sense to allow add_args that don't have to be
  # normalized by the split.
  # NOTE: Sequence requires special treatment because it has to be normalized in both
  # axes.
  train_add_args = dict( (k, v[task.train_indices]) for k,v in add_args.items())
  classifier =\
    learner(\
      task.train_vectors, 
      task.train_classes, 
      sequence=task.train_sequence,
      indices=task.train_indices,
      **train_add_args
      )

  test_add_args = dict( (k, v[task.test_indices]) for k,v in add_args.items())
  classifications =\
    classifier(\
      task.test_vectors, 
      sequence=task.test_sequence,
      indices=task.test_indices,
      **test_add_args
      )

  # Copy the metadata. Must ensure we do not pass a reference.
  metadata = {}
  metadata.update(classifier.metadata)

  result = result_from_task( task, classifications, metadata) 
  logger.debug(result)
  return result

class Experiment(object):
  #Produces a TaskSetResult
  def __init__(self, taskset, learner):
    self.taskset = taskset
    self.learner = learner 

  def run(self, add_args = None):
    raw_results =\
      [ run_task(self.learner, task, add_args=add_args) 
      for task 
      in ProgressIter(self.taskset.tasks, "Experiment: %s %s" % (self.learner.__name__, self.learner.params))
      ]
    metadata = dict(self.taskset.metadata)
    metadata['learner'] = self.learner.__name__
    metadata['learner_params'] = self.learner.params
    # TODO: is this the right place to generate uuid?
    metadata['uuid'] = uuid.uuid4()
    t = TaskSetResult(raw_results, metadata)
    return t


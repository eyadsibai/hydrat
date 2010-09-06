import time
import logging
import uuid
import numpy
from hydrat.result.result import result_from_task 
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.common.pb import ProgressIter

def run_task( learner, task ):
  logger = logging.getLogger('hydrat.experiment.run_task')
  classifier       = learner(task.train_vectors, task.train_classes)
  classifications  = classifier(task.test_vectors)

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

  def run(self):
    raw_results =\
      [ run_task(self.learner, task) 
      for task 
      in ProgressIter(self.taskset.tasks, "Experiment: %s %s" % (self.learner.__name__, self.learner.params))
      ]
    metadata = dict(self.taskset.metadata)
    metadata['learner'] = self.learner.__name__
    metadata['learner_params'] = self.learner.params
    metadata['uuid'] = uuid.uuid4()
    t = TaskSetResult(raw_results, metadata)
    t.metadata['avg_learn'] = numpy.mean(t.individual_metadata('learn_time'))
    t.metadata['avg_classify'] = numpy.mean(t.individual_metadata('classify_time'))
    return t


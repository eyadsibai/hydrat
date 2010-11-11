import numpy
import hydrat

from hydrat.frameworks.offline import OfflineFramework
from hydrat.result.result import Result
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.common.decorators import replace_with_result

import os
from hydrat.display.html import TableSort 
from hydrat.display.summary_fns import sf_featuresets
from hydrat.result.interpreter import SingleHighestValue, NonZero, SingleLowestValue
from hydrat.task.taskset import TaskSet
from hydrat.task.task import Task
# TODO: Allow for feature weighting and selection
# TODO: Produce tasksets, and bring this more in line with the offline framework.
#       This would reduce the burden in implementing weighting and selection.

import logging
logger = logging.getLogger(__name__)

summary_fields=\
  [ ( {'label':"Dataset", 'searchable':True}       , "dataset"       )
  , ( {'label':"Eval", 'searchable':True}       , "eval_dataset"       )
  , ( {'label':"Class Space",'searchable':True}     , "class_space"     )
  , ( {'label':"# Feats",'searchable':True}    , "num_featuresets"    )
  , ( {'label':"Feature Desc",'searchable':True}   , "feature_desc"     )
  , ( {'label':"Learner",'searchable':True}    , "learner"    )
  , ( {'label':"Params",'searchable':True}    , "learner_params"    )
  , ( "Macro-F"       , "macro_fscore"        )
  , ( "Macro-P"     , "macro_precision"     )
  , ( "Macro-R"        , "macro_recall"        )
  , ( "Micro-F"       , "micro_fscore"        )
  , ( "Micro-P"     , "micro_precision"     )
  , ( "Micro-R"        , "micro_recall"        )
  # TODO the next two are missing in the metadata somehow.
  #, ( {'sorter':'digit', 'label':"Learn Time"}    , "avg_learn"     )
  #, ( {'sorter':'digit', 'label':"Classify Time"} , "avg_classify"  )
  , ( {'sorter': None, 'label':"Details"}      , "link"          )
  ]

from hydrat.common.decorators import deprecated
class CrossDomainFramework(OfflineFramework):
  # We use the taskset interface to give us the ability to do weighting and
  # selection, just like in OfflineFramework. The approach here is basically
  # to build a taskset across the two datasets.
  
  def __init__(self, dataset, store=None):
    OfflineFramework.__init__(self, dataset, store)
    # TODO: Potentially set a different default summary function here!
    self.eval_dataset = None

  def set_eval_dataset(self, dataset):
    self.notify('Set eval_dataset to %s' % dataset.__name__)
    self.eval_dataset = dataset

  @property
  def taskset_desc(self):
    taskset_metadata = dict()
    taskset_metadata['dataset']         = self.dataset.__name__
    taskset_metadata['instance_space']  = self.dataset.instance_space
    taskset_metadata['split']           = self.split_name
    taskset_metadata['sequence']        = self.sequence_name
    taskset_metadata['feature_desc']    = self.feature_desc
    taskset_metadata['class_space']     = self.class_space
    taskset_metadata['task_type']       = self.__class__.__name__
    taskset_metadata['eval_dataset']    = self.eval_dataset.__name__
    taskset_metadata['eval_space']      = self.eval_dataset.instance_space
    return taskset_metadata

  @property
  def taskset(self):
    # TODO: Could save some additional recomputation by 
    # pulling the training featuremap/classmap from an already
    # saved version.
    md = self.taskset_desc
    if not self.store.has_TaskSet(md):
      self.notify('Generating TaskSet')
      # Ensure that the feature space has been processed for both datasets.
      # This is to avoid a synchronization issue resulting from the 'other'
      # dataset extending the space after the 'self' dataset has already read
      # its feature map.
      self.inducer.process_Dataset(self.dataset, fms=self.feature_spaces)
      self.inducer.process_Dataset(self.eval_dataset, fms=self.feature_spaces)

      other = OfflineFramework(self.eval_dataset, store = self.store) 
      other.set_feature_spaces(self.feature_spaces)
      other.set_class_space(self.class_space)

      task_md = dict(md)
      task_md['index'] = 0
      task = Task()
      task.train_vectors   = self.featuremap.raw
      task.train_classes   = self.classmap.raw
      task.train_sequence  = self.sequence
      task.train_indices   = self.train_indices

      task.test_vectors   = other.featuremap.raw
      task.test_classes   = other.classmap.raw
      task.test_sequence  = other.sequence
      task.test_indices   = other.train_indices

      task.metadata = task_md
      task.weights = {}

      taskset = TaskSet([task], md)
      self.store.new_TaskSet(taskset)
    return self.store.get_TaskSet(md, self.weights)

  @deprecated
  def evaluate(self, dataset):
    # TODO: How to handle feature weighting and selection?
    md = self.taskset_desc
    md['task_type'] = self.__class__.__name__
    md['eval_dataset'] = dataset.__name__
    md['eval_space'] = dataset.instance_space
    md.update(self.learner.metadata)
    if self.store.has_TaskSetResult(md):
      self.notify("Previously evaluated over '%s'" % dataset.__name__)
      return False 
    else:
      self.notify("Evaluating over '%s'" % dataset.__name__)
      other = Framework(dataset, store = self.store) 
      other.set_feature_spaces(self.feature_spaces)
      other.set_class_space(self.class_space)

      classifier = self.classifier
      # TODO: Need to transform the featuremap according to any transforms 
      #       they may have been applied! How??
      cl = classifier(other.featuremap.raw)
      md.update(classifier.metadata)

      gs = other.classmap.raw
      result = Result(gs, cl, numpy.arange(gs.shape[0]), md)
      tsr = TaskSetResult([result], md)
      self.store.new_TaskSetResult(tsr)
      return True

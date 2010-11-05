import logging
import os
import numpy
import scipy.sparse

import hydrat
import hydrat.display.summary_fns as sf
import hydrat.task.transform as tx
from hydrat.display.tsr import render_TaskSetResult
from hydrat.result.interpreter import SingleHighestValue, NonZero, SingleLowestValue
from hydrat.summary import classification_summary
from hydrat.display.summary_fns import sf_featuresets
from hydrat.display.store import results2html
from hydrat.common import as_set
from hydrat.preprocessor.features.transform import union
from hydrat.task.task import Task
from hydrat.task.taskset import TaskSet, from_partitions
from hydrat.experiments import Experiment
from hydrat.task.sampler import membership_vector
from hydrat.frameworks.common import Framework
from hydrat.common.decorators import deprecated

logger = logging.getLogger(__name__)

summary_fields=\
  [ ( {'label':"Dataset", 'searchable':True}       , "dataset"       )
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
  , ( {'sorter':'digit', 'label':"Learn Time"}    , "avg_learn"     )
  , ( {'sorter':'digit', 'label':"Classify Time"} , "avg_classify"  )
  , ( {'sorter': None, 'label':"Details"}      , "link"          )
  ]

class OfflineFramework(Framework):
  def __init__( self
              , dataset
              , store = None
              ):
    Framework.__init__(self, dataset, store)

    self.sequence_name = None
    self.outputP = None
    self.interpreter = SingleHighestValue()
    self.summary_fn = classification_summary()
    self.weights = [] # list of weights to be loaded with the taskset

  @property
  def classifier(self):
    raise NotImplementedError, "What should this actually do?"
    # We override the definition of classifier, as with a taskset context we want to
    # train the classifier over the taskset's training data, which may have had transforms
    # applied to it already.
    if self.learner is None:
      raise ValueError, "Learner has not been set"
    task = self.taskset.tasks[0]
    classifier = self.learner(task.train_vectors, task.train_classes, sequence=task.train_sequence)
    return classifier

  @property
  def summary(self):
    # TODO: Can we avoid loading the result?
    tsr_id = self.store._resolve_TaskSetResults(self.result_desc)[0]
    int_id = self.interpreter.__name__
    summary = self.store.get_Summary(tsr_id, int_id)
    missing_keys = set(self.summary_fn.keys) - set(summary)
    if len(missing_keys) > 0:
      result = self.result
      self.summary_fn.init(result, self.interpreter)
      new_values = dict( (key, self.summary_fn[key]) for key in missing_keys )
      self.store.add_Summary(tsr_id, int_id, new_values) 
      summary.update(new_values)
    return summary

  @property
  def result_desc(self):
    result_metadata = self.taskset_desc
    result_metadata['learner'] = self.learner.__name__
    result_metadata['learner_params'] = self.learner.params
    return result_metadata 

  @property 
  def result(self):
    if not self.store.has_TaskSetResult(self.result_desc):
      self.run()
    return self.store.get_TaskSetResult(self.result_desc)

  @property
  def taskset_desc(self):
    taskset_metadata = dict()
    taskset_metadata['dataset']         = self.dataset.__name__
    taskset_metadata['instance_space']  = self.dataset.instance_space
    taskset_metadata['split']           = self.split_name
    taskset_metadata['sequence']        = self.sequence_name
    taskset_metadata['feature_desc']    = self.feature_desc
    taskset_metadata['class_space']     = self.class_space
    return taskset_metadata

  @property
  def taskset(self):
    if not self.store.has_TaskSet(self.taskset_desc):
      self.notify('Generating TaskSet')
      # TODO: This is likely to break if not fully configured, so do something here.
      # We do this dance because we need to grab the featuremap and classmap under conditions
      # where we have no split, to get the full feature/classmaps. If this proves problematic,
      # should refactor.
      split = self.split
      split_name = self.split_name
      self.split_name = None
      fm = self.featuremap
      cm = self.classmap
      sq = self.sequence
      self.split_name = split_name
      taskset = from_partitions(split, fm, cm, sq, self.taskset_desc) 
      self.store.new_TaskSet(taskset)
    return self.store.get_TaskSet(self.taskset_desc, self.weights)

  def set_summary(self, summary_fn):
    self.summary_fn = summary_fn
    self.notify("Set summary_fn to '%s'" % summary_fn)

  def has_run(self):
    return self.store.has_TaskSetResult(self.result_desc)

  def run(self, add_args = None, force = False):
    # Check if we already have this result
    if force or not self.has_run():
      exp = Experiment(self.taskset, self.learner)
      try:
        tsr = exp.run(add_args=add_args)
        self.store.add_TaskSetResult(tsr)
        self.summary # forces the summary to be generated
      except Exception, e:
        logger.critical('Experiment failed with %s', e.__class__.__name__)
        if hydrat.config.getboolean('debug','pdb_on_classifier_exception'):
          logger.critical(e)
          import pdb;pdb.post_mortem()
        else:
          logger.debug(e)

  def transform_taskset(self, transformer, add_args=None):
    metadata = tx.update_metadata(self.taskset_desc, transformer)
    if not self.store.has_TaskSet(metadata):
      self.notify('Applying %s to taskset\n\t%s' % (str(transformer), str(self.taskset_desc)))
      self.weights = transformer.weights.keys()
      taskset = self.taskset
      new_taskset = tx.transform_taskset(taskset, transformer, add_args=add_args)
      self.store.extend_Weights(taskset)
      self.store.new_TaskSet(new_taskset)
    # Only copy over the new feature_desc
    self.feature_desc = metadata['feature_desc']

  def extend_taskset(self, feature_spaces):
    feature_spaces = as_set(feature_spaces)
    metadata = self.taskset_desc
    metadata['feature_desc'] += tuple(sorted(feature_spaces))
    if not self.store.has_TaskSet(self.taskset_desc):
      # Catch up any missing feature spaces
      self.inducer.process_Dataset(self.dataset, fms=feature_spaces)
      ds_name = self.dataset.__name__
      featuremaps = []
      for feature_space in sorted(feature_spaces):
        featuremaps.append(self.store.get_Data(ds_name, {'type':'feature','name':feature_space}))
      fm = union(*featuremaps)
      taskset = append_features(self.taskset, fm)
      self.store.new_TaskSet(taskset)
    self.feature_desc += tuple(sorted(feature_spaces))

  # TODO: Update to new-style summary function!!!
  def generate_output(self, path=None):
    """
    Generate HTML output
    """
    import sys
    sys.path.append('.')
    try:
      import browser_config
    except ImportError:
      import hydrat.browser.browser_config as browser_config

    if path is None: 
      path = hydrat.config.getpath('paths', 'output')
    if not os.path.exists(path): 
      os.mkdir(path)

    self.notify("Generating output")
    with open( os.path.join(path, 'index.html'), 'w' ) as f:
      f.write(results2html(self.store, browser_config))
    self.outputP = path

  def upload_output(self, target):
    """
    Copy output to a sepecified destination.
    Useful for transferring results to a webserver
    """
    self.notify("Uploading output to '%s'"% target)
    import updatedir
    updatedir.logger = logger
    updatedir.updatetree(self.outputP, target, overwrite=True)
    

from hydrat.display.html import TableSort 
@deprecated
def process_results( data_store
                   , result_store
                   , interpreter
                   , summary_fn=sf.sf_basic
                   , output_path=None
                   ):
  """
  If output_path is not None, per-result summaries will be produced in that folder.
  """
  summaries = []
  # TODO: Must only allow the framework to process results
  # relevant to it. Need to look into TaskSetResult metadata
  # to do this. 
  for resname in result_store._resolve_TaskSetResults({}):
    result = result_store._get_TaskSetResult(resname)
    summary = summary_fn(result, interpreter)
    summaries.append(summary)

    # If we are doing per-summary output
    if output_path is not None:
      resultpath_rel = os.path.join(output_path, str(result.metadata['uuid'])+'.html')
      if os.path.exists(resultpath_rel): 
        logger.debug("Not Reprocessing %s", resname)
      else:
        logger.debug("Processing %s --> %s",resname,resultpath_rel)
        with TableSort(open(resultpath_rel, 'w')) as result_renderer:
          result_renderer.section(resname) 
          class_space = data_store.get_Space(result.metadata['class_space'])
          render_TaskSetResult(result_renderer, result, class_space, interpreter, summary)
  return summaries
      

def append_features_update_metadata(metadata, fm):
  metadata = dict(metadata)
  metadata['feature_desc'] += fm.metadata['feature_desc']
  return metadata


def append_features(taskset, fm):
  #TODO
  raise NotImplementedError, "Need to update for sequence"
  new_tasks = []
  for task in taskset.tasks:
    assert len(task.train_indices) == len(task.test_indices) == fm.raw.shape[0]
    t = Task()
    t.train_indices = task.train_indices
    t.test_indices = task.test_indices
    t.train_classes = task.train_classes
    t.test_classes = task.test_classes

    # Extend vectors
    t.train_vectors = scipy.sparse.hstack((task.train_vectors,fm.raw[task.train_indices.nonzero()[0]])).tocsr()
    t.test_vectors = scipy.sparse.hstack((task.test_vectors,fm.raw[task.test_indices.nonzero()[0]])).tocsr()

    # Handle metadata
    t.metadata = append_features_update_metadata(task.metadata, fm)
    new_tasks.append(t)

  metadata = append_features_update_metadata(taskset.metadata, fm)
  ts = TaskSet(new_tasks, metadata)
  return ts

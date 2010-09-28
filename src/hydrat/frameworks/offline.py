import logging
import os
import numpy

import hydrat
import hydrat.display.summary_fns as sf
import hydrat.task.transform as tx
from hydrat.experiments import Experiment
from hydrat.display.tsr import render_TaskSetResult
from hydrat.result.interpreter import SingleHighestValue, NonZero, SingleLowestValue
from hydrat.display.html import TableSort 
from hydrat.preprocessor.model.inducer.dataset import DatasetInducer
from hydrat.store import Store, StoreError, NoData, AlreadyHaveData
from hydrat.display.summary_fns import sf_featuresets
from hydrat.display.html import TableSort 
from hydrat.display.tsr import result_summary_table
from hydrat.common.pb import ProgressIter
from hydrat.common import as_set
from hydrat.preprocessor.features.transform import union
from hydrat.task.task import Task
from hydrat.task.taskset import TaskSet, from_partitions
from hydrat.task.sampler import membership_vector
from hydrat.frameworks.common import init_workdir
import scipy.sparse

logger = logging.getLogger(__name__)

# TODO:
# set_feature_space should be able to deal with a list of feature spaces being passed.
# Ideally, it should receive a feature_desc object, but that is for further work.

summary_fields=\
  [ ( {'label':"Dataset", 'searchable':True}       , "dataset"       )
  , ( {'label':"Class Space",'searchable':True}     , "class_name"     )
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

class OfflineFramework(object):
  def __init__( self
              , dataset
              , work_path = None
              ):
    self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
    self.notify('Initializing')
    self.dataset = dataset
    self.work_path = work_path

    if work_path is None:
      generic_work_path = hydrat.config.get('paths','work')
      self.work_path = os.path.join(generic_work_path, self.__class__.__name__, dataset.__name__)
    
    init_workdir(self.work_path, ["output"])
    self.outputP  = os.path.join(self.work_path, 'output')
    self.store = Store(os.path.join(self.work_path,'store.h5'), 'a')
    self.inducer = DatasetInducer(self.store)

    self.feature_spaces = None
    self.class_space = None
    self.learner = None
    self.split_name = None
    self.sequence_name = None

    self.taskset_desc = None

  @property
  def taskset(self):
    return self.store.get_TaskSet(self.taskset_desc)

  @property
  def split(self):
    # TODO: grab from store instead. must ensure it has been induced.
    split_raw = self.dataset.split(self.split_name)
    if 'train' in split_raw and 'test' in split_raw:
      # Train/test type split.
      all_ids = self.dataset.instance_ids
      train_ids = membership_vector(all_ids, split_raw['train'])
      test_ids = membership_vector(all_ids, split_raw['test'])
      split = numpy.dstack((train_ids, test_ids)).swapaxes(0,1)
    elif any(key.startswith('fold') for key in split_raw):
      # Cross-validation folds
      all_ids = self.dataset.instance_ids
      folds_present = sorted(key for key in split_raw if key.startswith('fold'))
      partitions = []
      for fold in folds_present:
        test_ids = membership_vector(all_ids, split_raw[fold])
        train_docids = sum((split_raw[f] for f in folds_present if f is not fold), [])
        train_ids = membership_vector(all_ids, train_docids)
        partitions.append( numpy.dstack((train_ids, test_ids)).swapaxes(0,1) )
      split = numpy.hstack(partitions)
    else:
      raise ValueError, "Unknown type of split"
    return split

  @property
  def sequence(self):
    if self.sequence_name is None:
      return None
    else:
      return self.store.get_Sequence(self.dataset.__name__, self.sequence_name)

  @property
  def featuremap(self):
    ds_name = self.dataset.__name__
    featuremaps = []
    for feature_space in sorted(self.feature_spaces):
      featuremaps.append(self.store.get_Data(ds_name, {'type':'feature','name':feature_space}))

    # Join the featuremaps into a single featuremap
    fm = union(*featuremaps)
    return fm

  @property
  def classmap(self):
    ds_name = self.dataset.__name__
    return self.store.get_Data(ds_name, {'type':'class', 'name':self.class_space})

  def notify(self, str):
    self.logger.info(str)

  def set_feature_spaces(self, feature_spaces):
    self.inducer.process_Dataset( self.dataset, fms = feature_spaces)
    self.feature_spaces = as_set(feature_spaces)
    self.notify("Set feature_spaces to '%s'" % str(feature_spaces))
    self.configure()

  def set_class_space(self, class_space):
    self.inducer.process_Dataset( self.dataset, cms = class_space)
    self.class_space = class_space
    self.notify("Set class_space to '%s'" % class_space)
    self.configure()

  def set_sequence(self, sequence):
    self.inducer.process_Dataset( self.dataset, sqs = sequence)
    self.sequence_name = sequence
    self.notify("Set sequence to '%s'" % sequence)
    self.configure()

  def set_split(self, split):
    # TODO: Induce splits into Store
    self.notify("Setting split to '%s'" % split)
    self.split_name = split
    self.configure()

  def set_learner(self, learner):
    self.learner = learner
    self.notify("Set learner to '%s'" % learner)

  def is_configurable(self):
    return self.feature_spaces is not None\
      and self.class_space is not None\
      and self.split is not None

  def configure(self):
    if self.is_configurable():
      taskset_metadata = dict()
      taskset_metadata['dataset']       = self.dataset.__name__
      taskset_metadata['split']         = self.split_name
      taskset_metadata['sequence']      = self.sequence_name
      taskset_metadata['feature_desc']  = tuple(sorted(self.feature_spaces))
      taskset_metadata['class_space']   = self.class_space
      self.taskset_desc = taskset_metadata

  def has_run(self):
    #TODO: Fold part of this back into Store, in a has_TaskSetResult method
    m = dict( self.taskset_desc )
    m['learner'] = self.learner.__name__
    m['learner_params'] = self.learner.params
    taglist = self.store._resolve_TaskSetResults(m)
    logger.debug(m)
    logger.debug("%d previous results match this metadata", len(taglist))
    return len(taglist) > 0

  def run(self):
    # Check if we already have this result
    if not self.has_run():
      # Check if we already have this task
      if not self.store.has_TaskSet(self.taskset_desc):
        self.notify('Generating TaskSet')
        taskset = from_partitions(self.split, self.featuremap, self.classmap, self.sequence, self.taskset_desc) 
        self.store.new_TaskSet(taskset)
      run_experiment(self.taskset, self.learner, self.store)

  def process_tokenstream(self, tsname, extractor):
    dsname = self.dataset.__name__
    # Definition of space name.
    space_name = '_'.join((tsname,extractor.__name__))
    if not self.store.has_Data(dsname, space_name):
      self.notify("Inducing TokenStream '%s'" % tsname)
      # We always call this as if the ts has already been processed it is a fairly 
      # cheap no-op
      self.inducer.process_Dataset(self.dataset, tss=tsname)

      self.notify("Reading TokenStream '%s'" % tsname)
      tss = self.store.get_TokenStreams(dsname, tsname)
      instance_ids = self.store.get_InstanceIds(dsname)
      feat_dict = dict()
      for i, id in enumerate(ProgressIter(instance_ids, 'Processing %s' % extractor.__name__)):
        feat_dict[id] = extractor(tss[i])
      self.inducer.add_Featuremap(dsname, space_name, feat_dict)

  def transform_taskset(self, transformer):
    #TODO
    raise NotImplementedError, "Need to check if transformer needs update for sequence"
    metadata = tx.update_metadata(self.taskset_desc, transformer)
    if not self.store.has_TaskSet(metadata):
      taskset = tx.transform_taskset(self.taskset, transformer)
      self.store.new_TaskSet(taskset)
    self.taskset_desc = metadata

  def extend_taskset(self, feature_spaces):
    feature_spaces = as_set(feature_spaces)
    taskset_metadata = dict(self.taskset_desc)
    taskset_metadata['feature_desc'] += tuple(sorted(feature_spaces))
    if not self.store.has_TaskSet(taskset_metadata):
      # Catch up any missing feature spaces
      self.inducer.process_Dataset(self.dataset, fms=feature_spaces)
      ds_name = self.dataset.__name__
      featuremaps = []
      for feature_space in sorted(feature_spaces):
        featuremaps.append(self.store.get_Data(ds_name, {'type':'feature','name':feature_space}))
      fm = union(*featuremaps)
      taskset = append_features(self.taskset, fm)
      self.store.new_TaskSet(taskset)
    self.taskset_desc = taskset_metadata

  def generate_output(self, summary_fn=sf_featuresets, fields = summary_fields, interpreter = None):
    """
    Generate HTML output
    """
    self.notify("Generating output")
    #import tempfile
    #temp_output_path = tempfile.mkdtemp(dir=hydrat.config.getpath('paths','scratch'))
    summaries = process_results\
      ( self.store 
      , self.store
      , summary_fn = summary_fn
      , output_path= self.outputP
      , interpreter = interpreter
      ) 

    # render a HTML version of the summaries
    relevant = list(fields)
    for f_name in self.store.list_FeatureSpaces():
      relevant.append( ({'label':f_name, 'searchable':True}, 'feat_' + f_name) )

    indexpath = os.path.join(self.outputP, 'index.html')
    with TableSort(open(indexpath, "w")) as renderer:
      result_summary_table(summaries, renderer, relevant = relevant)

    #return temp_output_path

  def upload_output(self, target):
    """
    Copy output to a sepecified destination.
    Useful for transferring results to a webserver
    """
    self.notify("Uploading output to '%s'"% target)
    import updatedir
    updatedir.logger = logger
    updatedir.updatetree(self.outputP, target, overwrite=True)
    

def run_experiment(taskset, learner, result_store):
  exp = Experiment(taskset, learner)
  try:
    tsr = exp.run()
    result_store.add_TaskSetResult(tsr)
  except Exception, e:
    logger.critical('Experiment failed with %s', e.__class__.__name__)
    logger.debug(e)
    if hydrat.config.getboolean('debug','pdb_on_classifier_exception'):
      import pdb;pdb.post_mortem()

def process_results( data_store
                   , result_store
                   , summary_fn=sf.sf_basic
                   , output_path=None
                   , interpreter=None
                   ):
  """
  If output_path is not None, per-result summaries will be produced in that folder.
  """

  # Set a default interpreter
  if interpreter is None:
    interpreter = SingleHighestValue()

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

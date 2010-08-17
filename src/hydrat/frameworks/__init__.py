import logging
import os

import hydrat
import hydrat.display.summary_fns as sf
from hydrat.experiments import Experiment
from hydrat.display.tsr import render_TaskSetResult
from hydrat.result.interpreter import SingleHighestValue, NonZero, SingleLowestValue
from hydrat.display.html import TableSort 
from hydrat.preprocessor.model.inducer.dataset import DatasetInducer
from hydrat.store import open_store, UniversalStore, StoreError
from hydrat.display.summary_fns import sf_featuresets
from hydrat.display.html import TableSort 
from hydrat.display.tsr import result_summary_table

logger = logging.getLogger(__name__)

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

class Framework(object):
  def __init__( self
              , dataset
              , work_path = None
              , rng = None
              ):
    self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
    self.notify('Initializing')
    self.dataset = dataset
    self.work_path = work_path
    self.rng = rng

    if work_path is None:
      generic_work_path = hydrat.config.get('paths','work')
      self.work_path = os.path.join(generic_work_path, self.__class__.__name__, dataset.__name__)
    
    init_workdir(self.work_path, ["output"])
    self.outputP  = os.path.join(self.work_path, 'output')
    self.store = open_store(os.path.join(self.work_path,'store.h5'), 'a')

    self.feature_space = None
    self.class_space = None
    self.learner = None

  def notify(self, str):
    self.logger.info(str)

  def set_feature_space(self, feature_space):
    self.notify("Setting feature_space to '%s'" % feature_space)
    self.feature_space = feature_space
    self.configure()

  def set_class_space(self, class_space):
    self.notify("Setting class_space to '%s'" % class_space)
    self.class_space = class_space
    self.configure()

  def set_learner(self, learner):
    self.notify("Setting learner to '%s'" % learner)
    self.learner = learner

  def is_configurable(self):
    return self.feature_space is not None and self.class_space is not None

  def configure(self):
    if self.is_configurable():
      self.notify('Generating Model')
      self._generate_model()
      self.notify('Generating Partitioner')
      self.partitioner = self._generate_partitioner()
      self.notify('Generating Task')
      self.taskset = self._generate_taskset()

  def _generate_partitioner(self):
    raise NotImplementedError, "_generate_partitioner not implemented"

  def _generate_model(self):
    inducer = DatasetInducer(self.store)
    try:
      inducer.process_Dataset(self.dataset, self.feature_space, self.class_space)
    except StoreError, e:
      self.logger.debug(e)

  def _generate_taskset(self):
    ds_name = self.dataset.__name__
    fm = self.store.get_Data(ds_name, {'type':'feature','name':self.feature_space})
    # TODO
    # This is a stupid way of checking if we already have a taskset. We are building it anyway!!
    # The less stupid way would be to compute the full metadata, and check if that is in
    # the store already, then generate it if need be.
    taskset = self.partitioner(fm, {'name':'+'.join((self.feature_space, self.class_space))})
    try:
      self.store.new_TaskSet(taskset)
    except StoreError, e:
      pass
    return taskset

  def run(self):
    if self.feature_space is None:
      raise ValueError, "feature_space not yet set"
    if self.class_space is None:
      raise ValueError, "class_space not yet set"
    if self.learner is None:
      raise ValueError, "learner not yet set"
    run_experiment(self.taskset, self.learner, self.store)


  def generate_output(self):
    """
    .. todo:
      Allow the path to generate output to to be speficied. Or maybe use a file-like object
      The ultimate aim is to allow us to write to files on remote machines. Like hum!
      It can't be a file-like object because we need to write multiple files in a directory.
      Maybe an underlying sshfs mount could do the trick?
    """
    summaries = process_results\
      ( self.store 
      , self.store
      , summary_fn=sf_featuresets
      , output_path=self.outputP
      ) 

    # render a HTML version of the summaries
    relevant = list(summary_fields)
    for f_name in self.dataset.featuremap_names:
      relevant.append( ({'label':f_name, 'searchable':True}, 'feat_' + f_name) )

    indexpath = os.path.join(self.outputP, 'index.html')
    with TableSort(open(indexpath, "w")) as renderer:
      result_summary_table(summaries, renderer, relevant = relevant)

def init_workdir(path, newdirs=["models","tasks","results","output"]):
  """ Initialize the working directory, where various intermediate files will be stored.
  This is not to be considered a scratch folder, since the files stored here can be re-used.
  @param path The path to initialize
  """
  if os.path.exists(path):
    logger.warning('%s already exists', path)
  else:
    os.makedirs(path)
    for dir in newdirs:
      os.mkdir(os.path.join(path,dir))

def run_experiment(taskset, learner, result_store):
  keys = [ 'dataset_uuid'
         , 'feature_desc'
         , 'task_type'
         , 'rng_state'
         , 'class_uuid'
         ]
  m = dict( (k,taskset.metadata[k]) for k in keys )
  m['learner'] = learner.__name__
  m['learner_params'] = learner.params
  taglist = result_store._resolve_TaskSetResults(m)
  logger.debug(m)
  logger.debug("%d previous results match this metadata", len(taglist))
  if len(taglist) > 0:
    # Already did this experiment!
    logger.debug("Already have result; Not repeating experiment")
    return

  exp = Experiment(taskset, learner)
  try:
    tsr = exp.run()
    result_store.add_TaskSetResult(tsr)
  except Exception, e:
    logger.critical('Experiment failed with %s', e.__class__.__name__)
    logger.debug(e)
    #import pdb;pdb.post_mortem()

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
          class_space = data_store.get_Space(result.metadata['class_uuid'])
          render_TaskSetResult(result_renderer, result, class_space, interpreter, summary)
  return summaries
      

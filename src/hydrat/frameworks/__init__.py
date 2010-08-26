import logging
import os

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
    self.store = Store(os.path.join(self.work_path,'store.h5'), 'a')
    self.inducer = DatasetInducer(self.store)

    self.feature_spaces = None
    self.class_space = None
    self.learner = None

  def notify(self, str):
    self.logger.info(str)

  def set_feature_spaces(self, feature_spaces):
    # TODO: rescue code from tasks_combination to allow us to 
    #       combine feature spaces. We should be able to 
    #       handle a list of feature spaces (or set or tuple),
    #       which we then union to create a new feature space.
    #       This directly impacts the creation of TaskSets. We
    #       should automatically recognize a task that we already 
    #       have, in terms of the feature spaces contained in it.
    #       Do we care about the ordering of feature spaces? 
    #       Intuitively no, but practically it might make some difference.
    #       We could handle this by sorting first, or by treating the
    #       joined feature spaces as a set.
    #
    #       Receive via hydrat.common.as_set, so feature_space is always a set.
    #       Check that this doesn't break any derivin classes
    #       Fix the later plumbing to ensure that union is called if needed.

    self.feature_spaces = as_set(feature_spaces)
    self.notify("Set feature_spaces to '%s'" % feature_spaces)
    self.configure()

  def set_class_space(self, class_space):
    self.class_space = class_space
    self.notify("Set class_space to '%s'" % class_space)
    self.configure()

  def set_learner(self, learner):
    self.learner = learner
    self.notify("Set learner to '%s'" % learner)

  def process_tokenstream(self, tsname, ts_processor):
    dsname = self.dataset.__name__
    space_name = ts_processor.__name__
    if not self.store.has_Data(dsname, space_name):
      self.notify("Inducing TokenStream '%s'" % tsname)
      # We always call this as if the ts has already been processed it is a fairly 
      # cheap no-op
      self.inducer.process_Dataset(self.dataset, tss=tsname)

      self.notify("Reading TokenStream '%s'" % tsname)
      tss = self.store.get_TokenStreams(dsname, tsname)
      instance_ids = self.store.get_InstanceIds(dsname)
      feat_dict = dict()
      for i, id in enumerate(ProgressIter(instance_ids, 'Processing TokenStream')):
        feat_dict[id] = ts_processor(tss[i])
      self.inducer.add_Featuremap(dsname, space_name, feat_dict)

  def transform_taskset(self, transformer):
    metadata = tx.update_metadata(self.taskset, transformer)
    if self.store.has_TaskSet(metadata):
      # Load from store
      self.taskset = self.store.get_TaskSet(metadata)
    else:
      self.taskset = tx.transform_taskset(self.taskset, transformer)
      # Save to store
      try:
        self.store.new_TaskSet(self.taskset)
      except AlreadyHaveData:
        import pdb;pdb.set_trace()

  def is_configurable(self):
    return self.feature_spaces is not None and self.class_space is not None

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
    self.inducer.process_Dataset(self.dataset, fms=self.feature_spaces, cms=self.class_space)

  def _generate_taskset(self):
    ds_name = self.dataset.__name__
    featuremaps = []
    for feature_space in sorted(self.feature_spaces):
      featuremaps.append(self.store.get_Data(ds_name, {'type':'feature','name':feature_space}))

    # Join the featuremaps into a single featuremap
    fm = union(*featuremaps)

    additional_metadata = {}
    taskset_metadata = self.partitioner.generate_metadata(fm, additional_metadata)
    try:
      taskset = self.store.get_TaskSet(taskset_metadata)
    except NoData:
      taskset = self.partitioner(fm, additional_metadata)
      self.store.add_TaskSet(taskset)
    return taskset

  def run(self):
    if self.feature_spaces is None:
      raise ValueError, "feature_spaces not yet set"
    if self.class_space is None:
      raise ValueError, "class_space not yet set"
    if self.learner is None:
      raise ValueError, "learner not yet set"
    run_experiment(self.taskset, self.learner, self.store)

  def generate_output(self, summary_fn=sf_featuresets, fields = summary_fields):
    """
    Generate HTML output
    """
    #import tempfile
    #temp_output_path = tempfile.mkdtemp(dir=hydrat.config.getpath('paths','scratch'))
    summaries = process_results\
      ( self.store 
      , self.store
      , summary_fn = summary_fn
      , output_path= self.outputP
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
    import updatedir
    updatedir.logger = logger
    updatedir.updatetree(self.outputP, target, overwrite=True)
    

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
  keys = [ 'dataset'
         , 'feature_desc'
         , 'task_type'
         , 'rng_state'
         , 'class_space'
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
          class_space = data_store.get_Space(result.metadata['class_space'])
          render_TaskSetResult(result_renderer, result, class_space, interpreter, summary)
  return summaries
      

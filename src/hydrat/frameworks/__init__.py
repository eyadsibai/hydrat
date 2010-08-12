import logging
import os

import hydrat.display.summary_fns as sf
from hydrat.experiments import Experiment
from hydrat.display.tsr import render_TaskSetResult
from hydrat.result.interpreter import SingleHighestValue, NonZero, SingleLowestValue
from hydrat.display.html import TableSort 

logger = logging.getLogger(__name__)

class Framework(object):
  def notify(self, str):
    self.logger.info(str)

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
      

"""
This module provides a framework for evaluating multiple feature sets for a 
given classification task. It provides implementations of a variety of 
common operations carried out in this process, as well as a single
unifying method which only requires a dataset as a compulsory parameter, 
having default options for all other parameters.
"""
import os
import logging
from cPickle import dump
from itertools import combinations

import hydrat
import hydrat.display.summary_fns as sf
from hydrat.store import open_store, UniversalStore, StoreError
from hydrat.preprocessor.model.inducer.dataset import DatasetInducer
from hydrat.task.sampler import CrossValidate
from hydrat.experiments import Experiment
from hydrat.preprocessor.features.transform import union
from hydrat.result.interpreter import SingleHighestValue, NonZero, SingleLowestValue
from hydrat.display.html import TableSort 
from hydrat.display.tsr import result_summary_table, render_TaskSetResult
from hydrat.classifier import majorityL

logger = logging.getLogger(__name__)

def init_workdir(path):
  """ Initialize the working directory, where various intermediate files will be stored.
  This is not to be considered a scratch folder, since the files stored here can be re-used.
  @param path The path to initialize
  """
  if os.path.exists(path):
    logger.warning('%s already exists', path)
  else:
    os.makedirs(path)
    os.mkdir(os.path.join(path, 'models'))
    os.mkdir(os.path.join(path, 'tasks'))
    os.mkdir(os.path.join(path, 'results'))
    os.mkdir(os.path.join(path, 'output'))

def generate_model(store, datasets):
  """ Generate models for given datasets.
  A model is essentially a distribution over tokens. This function iterates over all 
  the listed datasets, and calls a DatasetInducer instance on each of them.
  The DatasetInducer instance saves the model into the given store
  """
  inducer = DatasetInducer(store)
  for ds in datasets:
    inducer(ds)

def generate_partitioning(data_store, ds, cm_name, folds=10, rng=None):
  """ Generate a partitioning of a particular dataset in a store.
  Note that this partitioning is done stratified with respect to a named
  class map. 
  @param data_store Store containing the relevant models
  @param ds The dataset we are interested in partitioning
  @param cm_name The name of the classmap used to stratify the partitioning
  @returns A function that when applied to ??? generates partitions
  """
  cv = CrossValidate(folds=folds, rng=rng)
  classmap = data_store.get_Data(ds, {'type':'class','name':cm_name})
  partitioner = cv(classmap)
  return partitioner
  
def tasks_combination( data_store
                     , task_store
                     , features 
                     , n
                     , ds
                     , partitioning
                     , rng=None
                     , baseline = None 
                     ):
  """
  Implements a system to combine different feature sets into tasks.
  """
  # n-ary combinations
  cm_name = partitioning.metadata['class_name']
  if baseline is None:
    fms = {}
  else:
    fms = {baseline:data_store.get_Data(ds, {'type':'feature','name':baseline})} 
  for fn in features: 
    fm = data_store.get_Data(ds, {'type':'feature','name':fn})
    fms[fn] = fm

  for fns in combinations(features, n):
    rel_fms = [fms[fn] for fn in fns]
    if baseline is not None:
      rel_fms.insert(0, fms[baseline])
    fm = union(*rel_fms)
    try:
      if baseline is None:
        task_store.new_TaskSet(partitioning(fm, {'name':'+'.join((cm_name,) +  fns)}))
      else:
        task_store.new_TaskSet(partitioning(fm, {'name':'+'.join((cm_name,baseline) +  fns)}))
    except StoreError,e:
      # This is how we discovere tasksets that already exist.
      # TODO: find an efficient way to do this! Constructing the whole thing is rather stupid
      logger.info(e)

def tasks_singlefeat( data_store
                    , task_store
                    , dataset
                    , task_classes
                    , features
                    , rng
                    ):
  """
  For each class, generate one task per feature set
  """
  for cm_name in task_classes:
    parts = generate_partitioning(data_store, dataset, cm_name, rng=rng)
    tasks_combination( data_store, task_store, features, 1, dataset, parts, rng=rng)

def tasks_ablation( data_store
                  , task_store
                  , dataset
                  , task_classes
                  , features
                  , rng
                  ):
  """
  For each class, generate one task per ablated feature set
  """
  for cm_name in task_classes:
    parts = generate_partitioning(data_store, dataset, cm_name, rng=rng)
    tasks_combination( data_store, task_store, features, len(features) - 1, dataset, parts, rng=rng)

def tasks_baseplus( data_store
                  , task_store
                  , dataset
                  , task_classes
                  , baseline_features
                  , added_features
                  , n
                  , rng
                  ):
  """
  For each class, generate tasks that represent augmentation of each
  baseline feature set with n feature sets from 'added_features'.
  """
  for cm_name in task_classes:
    parts = generate_partitioning(data_store, dataset, cm_name, rng=rng)
    for baseline in baseline_features:
      tasks_combination( data_store, task_store, added_features, n, dataset, parts, baseline=baseline, rng=rng)


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
  logger.info("%d previous results match this metadata", len(taglist))
  if len(taglist) > 0:
    # Already did this experiment!
    logger.info("Already have result; Not repeating experiment")
    return

  exp = Experiment(taskset, learner)
  try:
    tsr = exp.run()
    result_store.add_TaskSetResult(tsr)
  except Exception, e:
    logger.critical('Experiment failed with %s', e.__class__.__name__)
    logger.debug(e.message)
    import pdb;pdb.post_mortem()

def run_tasksets(tasksets, result_store, learners):
  for taskset in tasksets:
    for l in learners:
      try:
        run_experiment(taskset, l, result_store)
      except Exception, e:
        logger.critical('Experiment failed!')
        logger.critical(e)
        import pdb;pdb.post_mortem()

def load_tasksets(task_store, task_meta={}):
  return [task_store._get_TaskSet(tag) for tag in task_store._resolve_TaskSet(task_meta)]

def run_experiments(task_store, result_store, learners, task_meta={}):
  tasksets = load_tasksets(task_store, task_meta)
  run_tasksets(tasksets, result_store, learners)


def establish_baseline(task_store, result_store, task_classes):
  learners = [ majorityL() ]
  for c in task_classes:
    tasksets = load_tasksets(task_store, dict(name='+'.join((c,'bag_of_words'))))
    run_tasksets(tasksets, result_store, learners)

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
        logger.info("Not Reprocessing %s", resname)
      else:
        logger.info("Processing %s --> %s",resname,resultpath_rel)
        with TableSort(open(resultpath_rel, 'w')) as result_renderer:
          result_renderer.section(resname) 
          class_space = data_store.get_Space(result.metadata['class_uuid'])
          render_TaskSetResult(result_renderer, result, class_space, interpreter, summary)
  return summaries
      
def default_crossvalidation\
  ( dataset
  , work_path = None
  , baseline_features = None
  , added_features = None
  , task_classes = None
  , learners = None
  , rng = None
  ):
  """
  .. todo:
      Could rearrange operations to produce output as it is available,
      rather than doing all steps in sequence.
  """

  ####
  ## Use default values for parameters which were not user-specified
  ####

  # Set up a work path in the scratch directory according to the
  # dataset name if None is provided.
  if work_path is None:
    work_path = os.path.join(hydrat.config.get('paths','work'), 'crossvalidation', dataset.__name__)

  if baseline_features is None:
    baseline_features = []

  # Compute the added features, removing any baseline features that have been
  # specified.
  if added_features is None:
    features = set(dataset.featuremap_names)
    added_features = list(features - set(baseline_features))

  if task_classes is None:
    task_classes = list(dataset.classmap_names)

  if learners is None:
    from hydrat.classifier.SVM import bsvmL
    learners =\
      [ bsvmL(kernel_type='linear')
      ]

  if rng is None:
    rng = hydrat.rng

  # Compute the full set of features
  all_features = baseline_features + added_features

  # Initialize the working directory
  init_workdir(work_path)
  modelP   = os.path.join(work_path, 'models')
  taskP    = os.path.join(work_path, 'tasks')
  resultP  = os.path.join(work_path, 'results')
  outputP  = os.path.join(work_path, 'output')


  ####
  ## Process the raw data, calculating all feature values.
  ####
  data = open_store(os.path.join(modelP,'model.h5'), 'a')
  ds_name = dataset.__name__
  generate_model(data, [dataset])

  ####
  ## Set up Tasks
  ###
  task_files = []
  data_store = open_store(os.path.join(modelP,'model.h5'))
  # TODO: selection of which feature combinations to use
  
  # Each featureset in isolation
  task_store = open_store(os.path.join(taskP, 'singlefeat.h5'), 'a')
  tasks_singlefeat(data_store, task_store, ds_name, task_classes, all_features, rng)
  task_files.append('singlefeat.h5')

  # Feature ablation
  task_store = open_store(os.path.join(taskP, 'singlefeat.h5'), 'a')
  tasks_ablation(data_store, task_store, ds_name, task_classes, all_features, rng)
  task_files.append('singlefeat.h5')

  if baseline_features is not None:
    # Baseline + 1 feature set
    task_store = open_store(os.path.join(taskP, 'baseplusone.h5'), 'a')
    tasks_baseplus(data_store, task_store, ds_name, task_classes, baseline_features, added_features, 1, rng)
    task_files.append('baseplusone.h5')

    # Baseline + 2 feature sets
    task_store = open_store(os.path.join(taskP, 'baseplustwo.h5'), 'a')
    tasks_baseplus(data_store, task_store, ds_name, task_classes, baseline_features, added_features, 2, rng)
    task_files.append('baseplustwo.h5')


  ####
  ## Run Experiments
  ####
  task = open_store(os.path.join(taskP, 'singlefeat.h5'))
  result_files = []
  
  # Get baseline results from a majority class learner
  result = open_store(os.path.join(resultP,'baseline.h5'), 'a')
  establish_baseline(task, result, task_classes)
  result_files.append('baseline.h5')

  # Run the experiment over each learner, for each class labelling specified
  result = open_store(os.path.join(resultP, 'result.h5'), 'a')
  result_files.append('result.h5')
  for f in task_files:
    task = open_store(os.path.join(taskP,f))
    for c in task_classes:
      tasksets = load_tasksets(task, dict(class_name=c))
      run_tasksets(tasksets, result, learners)

  ####
  ## Produce output
  ####
  # compute per-result summaries
  summary_fn = sf.sf_featuresets
  summaries = []
  for r in result_files: 
    result_store = open_store(os.path.join(resultP,r))
    summaries.extend( process_results( data_store, result_store, summary_fn=summary_fn, output_path=outputP) )

  # write summaries to a pickle file 
  with open(os.path.join(outputP,'summaries.pickle'),'w') as f:
    dump(summaries,f)

  # render a HTML version of the summaries
  relevant = \
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
  for f_name in all_features:
    relevant.append( ({'label':f_name, 'searchable':True}, 'feat_' + f_name) )

  indexpath = os.path.join(outputP, 'index.html')
  with TableSort(open(indexpath, "w")) as renderer:
    result_summary_table(summaries, renderer, relevant = relevant)


"""
This framework provides an implementation of a single-class approach to carrying out
a cross-validation experiment in hydrat.
"""
import logging
import os

import hydrat
from hydrat.store import open_store, UniversalStore, StoreError
from hydrat.preprocessor.model.inducer.dataset import DatasetInducer
from hydrat.task.sampler import CrossValidate
from hydrat.display.summary_fns import sf_featuresets
from hydrat.display.html import TableSort 
from hydrat.display.tsr import result_summary_table
from . import init_workdir, Framework, run_experiment, process_results

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

class CrossValidation(Framework):
  def __init__\
    ( self
    , dataset
    , work_path = None
    , folds = 10
    , rng = hydrat.rng
    ):
    self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
    self.notify('Initializing')
    self.dataset = dataset
    self.work_path = work_path
    self.folds = folds
    self.rng = rng

    if work_path is None:
      generic_work_path = hydrat.config.get('paths','work')
      self.work_path = os.path.join(generic_work_path, 'crossvalidation', dataset.__name__)
    
    init_workdir(self.work_path, ["output"])
    self.outputP  = os.path.join(self.work_path, 'output')
    self.store = open_store(os.path.join(self.work_path,'store.h5'), 'a')

    self.feature_space = None
    self.class_space = None

  def set_feature_space(self, feature_space):
    self.notify("Setting feature_space to '%s'" % feature_space)
    self.feature_space = feature_space
    if self.class_space is not None:
      self.configure()

  def set_class_space(self, class_space):
    self.notify("Setting class_space to '%s'" % class_space)
    self.class_space = class_space
    if self.feature_space is not None:
      self.configure()

  def configure(self):
    self.notify('Generating Model')
    self._generate_model()
    self.notify('Generating Partitioner')
    self.partitioner = self._generate_partitioner()
    self.notify('Generating Task')
    self.taskset = self._generate_taskset()

  def _generate_model(self):
    inducer = DatasetInducer(self.store)
    try:
      inducer.process_Dataset(self.dataset, self.feature_space, self.class_space)
    except StoreError, e:
      self.logger.debug(e)

  def _generate_partitioner(self):
    cv = CrossValidate(folds=self.folds, rng=self.rng)
    ds_name = self.dataset.__name__
    classmap = self.store.get_Data(ds_name, {'type':'class','name':self.class_space})
    partitioner = cv(classmap)
    return partitioner

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

  def run_learner(self, learner):
    if self.feature_space is None:
      raise ValueError, "feature_space not yet set"
    if self.class_space is None:
      raise ValueError, "class_space not yet set"
    run_experiment(self.taskset, learner, self.store)


  def generate_output(self):
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

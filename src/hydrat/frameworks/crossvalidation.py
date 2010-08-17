"""
This framework provides an implementation of a single-class approach to carrying out
a cross-validation experiment in hydrat.
"""
import logging
import os

import hydrat
from hydrat.task.sampler import CrossValidate
from hydrat.frameworks import init_workdir, Framework, run_experiment, process_results, summary_fields

class CrossValidationFramework(Framework):
  def __init__\
    ( self
    , dataset
    , work_path = None
    , folds = 10
    , rng = hydrat.rng
    ):
    Framework.__init__(self, dataset, work_path, rng)
    self.folds = folds

  def _generate_partitioner(self):
    cv = CrossValidate(folds=self.folds, rng=self.rng)
    ds_name = self.dataset.__name__
    classmap = self.store.get_Data(ds_name, {'type':'class','name':self.class_space})
    partitioner = cv(classmap)
    return partitioner


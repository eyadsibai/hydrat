"""
This is a general framework for using hydrat for the purposes of
doing classification over train/test splits of data. 

It is very similar to (TODO: based on??) the novel instances framework,
the key difference being that here we have the goldstandard class
labels of the test instances, so we can analyze performance.

..todo:
  - this framework must be able to co-operate with novel_instances, to
  	allow us to participate in a shared task, then to evaluate our 
  	performance when goldstandard results are distributed.
"""
import logging
import os
import numpy

import hydrat
from hydrat.frameworks import Framework
from hydrat.task.sampler import membership_vector, PresetSplit

class PresetSplitFramework(Framework):
  def __init__\
    ( self
    , dataset
    , work_path = None
    , rng = hydrat.rng
    ):
    Framework.__init__(self, dataset, work_path, rng)
    self.split = None

  def set_split(self, split):
    self.notify("Setting split to '%s'" % split)
    split_raw = self.dataset.split(split)
    all_ids = self.dataset.instance_ids
    train_ids = membership_vector(all_ids, split_raw['train'])
    test_ids = membership_vector(all_ids, split_raw['test'])
    self.split = numpy.dstack((train_ids, test_ids)).swapaxes(0,1)
    self.configure()

  def is_configurable(self):
    return self.feature_space is not None\
      and self.class_space is not None\
      and self.split is not None

  def _generate_partitioner(self):
    ps = PresetSplit(self.split, rng=self.rng)
    ds_name = self.dataset.__name__
    classmap = self.store.get_Data(ds_name, {'type':'class','name':self.class_space})
    partitioner = ps(classmap)
    return partitioner

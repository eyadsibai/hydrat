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

import hydrat
from hydrat.frameworks import Framework
from hydrat.task.sampler import TrainTest

class TrainTestFramework(Framework):
  def __init__\
    ( self
    , dataset
    , ratio = 4
    , work_path = None
    , rng = hydrat.rng
    ):
    Framework.__init__(self, dataset, work_path, rng)
    self.ratio = ratio

  def _generate_partitioner(self):
    tt = TrainTest(ratio=self.ratio, rng=self.rng)
    ds_name = self.dataset.__name__
    classmap = self.store.get_Data(ds_name, {'type':'class','name':self.class_space})
    partitioner = tt(classmap)
    return partitioner

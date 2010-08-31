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
    # Split name is important because it is the only way to distinguish tasks via metadata
    self.split_name = None

  def set_split(self, split):
    # TODO: Work out the type of preset split!!!
    self.notify("Setting split to '%s'" % split)
    self.split_name = split
    split_raw = self.dataset.split(split)
    if 'train' in split_raw and 'test' in split_raw:
      # Train/test type split.
      all_ids = self.dataset.instance_ids
      train_ids = membership_vector(all_ids, split_raw['train'])
      test_ids = membership_vector(all_ids, split_raw['test'])
      self.split = numpy.dstack((train_ids, test_ids)).swapaxes(0,1)
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
      self.split = numpy.hstack(partitions)
    else:
      raise ValueError, "Unknown type of split"
    self.configure()

  def is_configurable(self):
    return self.feature_spaces is not None\
      and self.class_space is not None\
      and self.split is not None

  def _generate_partitioner(self):
    md = {'split_name': self.split_name }
    ps = PresetSplit(self.split, metadata=md,rng=self.rng)
    ds_name = self.dataset.__name__
    classmap = self.store.get_Data(ds_name, {'type':'class','name':self.class_space})
    partitioner = ps(classmap)
    return partitioner

"""
This module contains the external classification framework, which bypasses 
hydrat's internal feature management mechanisms, allowing the external packages
themselves to provide this functionality. Essentially, what you get is wrappers
that take TextDataset objects, and produce standard Result objects, which can
then be evaluated using all the usual metrics, and have their results included 
with results from the hydrat-managed classifications.
"""
import time
import random
import numpy
import logging
import datetime as dt
from itertools import izip
from hydrat.task.sampler import allocate, stratify
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.result.interpreter import SingleHighestValue
from hydrat.result.result import Result

def select(seq, bools):
  return [d for d,c in izip(seq, bools) if c]

class ExternalTask(object):
  def __init__(self, train_data, train_class, test_data, test_class, instance_indices, metadata):
    self.train_data = train_data
    self.test_data = test_data
    self.train_class = train_class
    self.test_class = test_class
    self.instance_indices = instance_indices
    self.metadata = metadata

class ExternalCrossValidate(object):
  def __init__(self, data, classmap, folds = 10, seed = None, metadata = {}):
    self.data = data
    self.classmap = classmap

    if seed is None:
      seed = time.time()

    self.seed = seed
    self.rng = random.Random()
    self.rng.seed(seed)

    self.metadata = {}
    self.metadata.update(metadata)
    self.metadata['seed'] = seed

    strata_map = stratify(classmap)
    partition_proportions = numpy.array([1] * folds )
    self.folds = allocate( strata_map
                         , partition_proportions
                         , probabilistic = False
                         , rng=self.rng
                         ) 
  def tasks(self):
    tasklist = []
    for i in range(self.folds.shape[1]):
      fold_row   = self.folds[:,i]
      test_ids   = fold_row
      train_ids  = numpy.logical_not(fold_row)
      md = {}
      md.update(self.metadata)
      tasklist.append( 
        ExternalTask( select(self.data, train_ids)
                    , self.classmap[train_ids]
                    , select(self.data, test_ids)
                    , self.classmap[test_ids]
                    , select(range(len(test_ids)),test_ids)
                    , md
                    ) 
                     )
    return tasklist
    
class ExternalExperiment(object):
  def __init__(self, taskset, classif):
    self.taskset = taskset
    self.classif = classif

  def run(self):
    cv = self.taskset
    classif = self.classif
    results = []
    for t in cv.tasks():
      c = classif()
      c.train(t.train_data, t.train_class)
      cl = c.classify(t.test_data)
      gs = t.test_class
      ii = t.instance_indices
      md = {}
      md.update(t.metadata) 
      md.update({'learn_time':c.train_time, 'classify_time':c.classify_time})
      r = Result(gs,cl,ii,md)
      results.append(r)
      print r.classification_matrix(SingleHighestValue())
    metadata = {}
    metadata['date'] = dt.datetime.now().isoformat()
    metadata['classifier'] = classif.__name__
    metadata['seed'] = self.taskset.seed
    return TaskSetResult(results, metadata)

#Closely modelled on examples.sample_data.RunCrossvalidation
def process(store, cache, classif, ds, class_name, seed = None, repeat = False):
  logger = logging.getLogger('hydrat.external.process')

  space_meta = {'type':'class', 'name':class_name}
  space_tag = store.resolve_Space(space_meta)
  if not cache.has_Space(space_tag):
    space = store.get_Space(space_tag)
    cache.add_Space(space, space_meta, space_tag) 

  ds_tag = store.resolve_Dataset(ds.__name__)
  class_map = store.get_ClassData(ds_tag, space_tag)

  text = ds.text()
  data = [ text[i] for i in store.instance_identifiers(ds_tag) ]

  tsr_metadata = { 'dataset' : ds.__name__
                 , 'class_name' : class_name
                 , 'class_uuid' : space_tag 
                 , 'feature_name' : 'EXTERNAL'
                 , 'dataset_uuid' : ds_tag
                 , 'classifier' : classif.__name__
                 , 'seed' : seed
                 }
  taglist = cache.resolve_tag(tsr_metadata)
  if not repeat and len(taglist) > 0:
    logger.info("Already have result, not repearting experiment")
    return cache[taglist[0]]

  x = ExternalCrossValidate(data, class_map, metadata = tsr_metadata)
  e = ExternalExperiment(x, classif)
  tsr = e.run()
  
  cache.add_TaskSetResult(tsr, tsr_metadata)
  return tsr


"""
Demonstrating parallelization of experiment folds
"""

import hydrat.corpora.dummy as dummy
from hydrat.dataset.split import CrossValidation

import hydrat.classifier.naivebayes as nb
import hydrat.classifier.liblinear as liblinear

from hydrat.proxy import DataProxy
from hydrat.experiment import Experiment

class unicode_dummy(dummy.unicode_dummy, CrossValidation): pass


if __name__ == "__main__":
  ds = unicode_dummy(100)
  proxy = DataProxy(ds)
  proxy.class_space = 'dummy_default'
  proxy.feature_spaces = 'byte_bigram'
  proxy.split_name = 'crossvalidation'

  l = nb.multinomialL()
  ts = proxy.taskset
  e = Experiment(ts, l, parallel=False)
  r1 = e.results
  e = Experiment(ts, l, parallel=True)
  r2 = e.results
  print r1 == r2


"""
Example for learning curves in a basic in-domain 
text classification task
"""

import hydrat.corpora.dummy as dummy
from hydrat.dataset.split import LearnCurve

import hydrat.classifier.naivebayes as nb

from hydrat.proxy import DataProxy
from hydrat.experiment import Experiment

class unicode_dummy(dummy.unicode_dummy, LearnCurve): pass

if __name__ == "__main__":
  ds = unicode_dummy(100)
  proxy = DataProxy(ds)
  proxy.class_space = 'dummy_default'
  proxy.feature_spaces = 'byte_bigram'
  proxy.split_name = 'learncurve'

  l = nb.multinomialL()
  e = Experiment(proxy, l)
  proxy.store.new_TaskSetResult(e)


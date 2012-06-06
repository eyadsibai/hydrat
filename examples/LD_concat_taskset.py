"""
Example for using ConcatTaskSet to apply feature weight
recycling for LD feature selection in composited feature spaces.

Marco Lui, June 2012
"""
import hydrat
from hydrat.corpora.dummy import unicode_dummy

from hydrat.proxy import DataProxy, InductiveLOO
from hydrat.store import Store
from hydrat.experiment import Experiment

import hydrat.classifier.naivebayes as nb
import hydrat.classifier.nearest_prototype as np
import hydrat.classifier.baseline as ba
import hydrat.classifier.libsvm as svm
import hydrat.classifier.naivebayes as nb
import hydrat.classifier.liblinear as liblinear

from hydrat.transformer import Transform
from hydrat.transformer.featureselect import term_count_top, LangDomain
from hydrat.task.concat import ConcatTaskSet


datasets = [
  unicode_dummy(10),
  unicode_dummy(20),
  unicode_dummy(30),
  unicode_dummy(40),
  ]

learners=[
  #liblinear.liblinearL(svm_type=2),
  #np.skew_mean_prototypeL(),
  nb.multinomialL(),
  ]

features=[
  "byte_bigram",
  "byte_trigram",
  "byte_unigram",
  ]


if __name__ == "__main__":

  store = Store.from_caller()

  x = []

  for feature in features:
    proxies = []
    for ds in datasets:
      proxy = DataProxy(ds, store=store)
      proxy.class_space = 'dummy_default'
      proxy.feature_spaces = feature
      proxies.append(proxy)
    p = InductiveLOO(proxies)
    dm = p.domainmap
    ts = store.new_TaskSet(p)
    Transform(ts, LangDomain(dm)).tasks
    x.append(ts)

  y = ConcatTaskSet(x)
  z = store.new_TaskSet(y)
  ts = Transform(z, LangDomain(dm))
  l = nb.multinomialL()
  e = Experiment(ts, l)
  r = store.new_TaskSetResult(e)
  import pdb;pdb.set_trace() 

"""
Demonstrating parallelization of experiment folds
"""

import hydrat.corpora.dummy as dummy
from hydrat.dataset.split import CrossValidation
from hydrat.dataset.text import ByteQuadgram, BytePentagram

import hydrat.classifier.liblinear as liblinear
import hydrat.classifier.nearest_prototype as np
import hydrat.classifier.naivebayes as nb

from hydrat.proxy import DataProxy
from hydrat.experiment import Experiment
import hydrat.corpora.langid.naacl2010 as naacl2010

class EuroGOV(naacl2010.EuroGOV, CrossValidation, ByteQuadgram, BytePentagram): pass

import hydrat.browser.browser_config as browser_config

if __name__ == "__main__":
  ds = EuroGOV()
  proxy = DataProxy(ds)
  proxy.class_space = 'iso639_1'
  proxy.feature_spaces = 'byte_quadgram'
  proxy.split_name = 'crossvalidation'

  #l = liblinear.liblinearL(svm_type=2)
  l = nb.multinomialL()
  ts = proxy.taskset
  e = Experiment(ts, l, parallel=True)
  tsr = proxy.store.new_TaskSetResult(e)
  summary_fn = browser_config.summary_fn
  interpreter = browser_config.interpreter
  tsr.summarize(summary_fn, interpreter)
  print "P:{micro_precision} R:{micro_recall} F:{micro_fscore}".format(**tsr.summaries['SingleHighestValue'])


import hydrat.corpora.langid.naacl2010 as naacl

import hydrat.classifier.nearest_prototype as np
import hydrat.classifier.maxent as maxent
import hydrat.classifier.SVM as svm
import hydrat.classifier.knn as knn

from hydrat.proxy import DataProxy
from hydrat.store import Store
from hydrat.experiment import Experiment

learners = [ 
  np.cosine_mean_prototypeL(),
  np.skew_mean_prototypeL(),
  knn.skew_1nnL(),
  knn.oop_1nnL(),
  knn.cosine_1nnL(),
  maxent.maxentLearner(),
  svm.libsvmExtL(kernel_type='linear'),
  svm.libsvmExtL(kernel_type='rbf'),
  ]

feature_spaces = [
  'codepoint_unigram',
  'byte_unigram',
  'codepoint_bigram',
  'byte_bigram',
  'codepoint_trigram',
  'byte_trigram',
  ]

datasets = [
  naacl.EuroGOV(),
  naacl.TCL(),
  naacl.Wikipedia(),
  ]

if __name__ == "__main__":
  store = Store.from_caller()

  for ds in datasets:
    proxy = DataProxy(ds, store=store)
    proxy.class_space = 'iso639_1'
    for feature in feature_spaces:
      proxy.feature_spaces = feature

      for l in learners:
        e = Experiment(proxy, l)
        r = store.new_TaskSetResult(e)

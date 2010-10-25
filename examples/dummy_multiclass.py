from hydrat.frameworks.offline import OfflineFramework
import hydrat.corpora.dummy as dummy

from hydrat.dataset.split import CrossValidation 
class unicode_dummy_multiclass(dummy.unicode_dummy_multiclass, CrossValidation): pass

import hydrat.classifier.nearest_prototype as np
import hydrat.classifier.knn as knn
import hydrat.classifier.meta.binary as binary
import hydrat.classifier.meta.stratified as stratified

learners=\
  [ np.cosine_mean_prototypeL()
  , knn.cosine_1nnL()
  ]

if __name__ == "__main__":
  fw = OfflineFramework(unicode_dummy_multiclass())
  fw.set_class_space('dummy_multiclass')
  fw.set_feature_spaces('byte_unigram')
  fw.set_split('crossvalidation')
  for l in learners:
    fw.set_learner(l)
    fw.run()
    fw.set_learner(binary.BinaryLearner(l))
    fw.run()
    fw.set_learner(stratified.StratifiedLearner(l))
    fw.run()

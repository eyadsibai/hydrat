from hydrat.frameworks.offline import OfflineFramework
import hydrat.corpora.dummy as dummy

# Subclass dummy.unicode_dummy, to add an automatically-constructed
# train/test split to it, as well as an automatically-contrsucted
# 10-fold cross-validation split.
from hydrat.dataset.split import TrainTest, CrossValidation
class unicode_dummy(dummy.unicode_dummy, TrainTest, CrossValidation): pass

import hydrat.classifier.NLTK as nltk
import hydrat.classifier.SVM as svm
import hydrat.classifier.knn as knn
import hydrat.classifier.nearest_prototype as np
import hydrat.classifier.weka as weka
import hydrat.classifier.maxent as maxent

learners=\
  [ knn.cosine_1nnL()
  , nltk.naivebayesL()
  , nltk.decisiontreeL()
  , svm.libsvmExtL(kernel_type='linear')
  , svm.bsvmL(kernel_type='linear')
  , knn.skew_1nnL()
  , knn.oop_1nnL()
  , np.cosine_mean_prototypeL()
  , maxent.maxentLearner()
  , weka.majorityclassL()
  , weka.nbL()
  , weka.j48L()
  ]
if __name__ == "__main__":
  fw = OfflineFramework(unicode_dummy())
  fw.set_class_space('dummy_default')
  fw.set_feature_spaces('byte_unigram')

  # Run over train/test split
  fw.set_split('traintest')
  for l in learners:
    fw.set_learner(l)
    fw.run()

  # Run over crossvalidation split
  fw.set_split('crossvalidation')
  split = fw.split
  for l in learners:
    fw.set_learner(l)
    fw.run()

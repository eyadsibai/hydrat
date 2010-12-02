from hydrat.frameworks.offline import OfflineFramework

import hydrat.corpora.reuters as reuters

import hydrat.classifier.NLTK as nltk
import hydrat.classifier.SVM as svm
import hydrat.classifier.knn as knn
import hydrat.classifier.nearest_prototype as np
import hydrat.classifier.weka as weka
import hydrat.classifier.maxent as maxent

if __name__ == "__main__":
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
  fw = OfflineFramework(reuters.Reuters21578())
  fw.set_class_space('reuters21578_topics')
  fw.set_feature_spaces('word_unigram')
  fw.set_split('lewis')
  for l in learners:
    fw.set_learner(l)
    fw.run()

import hydrat.corpora.langid.naacl2010 as naacl

import hydrat.classifier.nearest_prototype as np
import hydrat.classifier.maxent as maxent
import hydrat.classifier.SVM as svm
import hydrat.classifier.knn as knn
from hydrat.frameworks.offline import OfflineFramework 

learners=\
  [ np.cosine_mean_prototypeL()
  , np.skew_mean_prototypeL()
  , knn.skew_1nnL()
  , knn.oop_1nnL()
  , knn.cosine_1nnL()
  , maxent.maxentLearner()
  , svm.libsvmExtL(kernel_type='linear')
  , svm.libsvmExtL(kernel_type='rbf')
  ]

feature_spaces=\
  [ 'codepoint_unigram'
  , 'byte_unigram'
  , 'codepoint_bigram'
  , 'byte_bigram'
  , 'codepoint_trigram'
  , 'byte_trigram'
  ]

datasets=\
  [ naacl.EuroGOV()
  , naacl.TCL()
  , naacl.Wikipedia()
  ]

if __name__ == "__main__":
  for ds in datasets:
    fw = OfflineFramework(ds, store='./naacl2010' )
    fw.set_class_space('iso639_1')
    fw.set_split('crossvalidation')
    for fs in feature_spaces:
      fw.set_feature_spaces(fs)
      for l in learners:
        fw.set_learner(l)
        fw.run()
    fw.generate_output()


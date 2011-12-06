from hydrat.frameworks.offline import OfflineFramework

from hydrat.corpora.langid.altw2010 import ALTW2010

import hydrat.classifier.SVM as svm
import hydrat.classifier.knn as knn
import hydrat.classifier.nearest_prototype as np
import hydrat.classifier.maxent as maxent
import hydrat.classifier.baseline as baseline

if __name__ == "__main__":
  learners=\
    [ np.skew_mean_prototypeL()
    , knn.skew_1nnL()
    , baseline.majorityL()
    , svm.libsvmExtL(kernel_type='linear')
    , svm.libsvmExtL(kernel_type='rbf')
    ]
  features=\
    [ 'byte_unigram'
    , 'codepoint_unigram'
    , 'byte_bigram'
    , 'codepoint_bigram'
    , 'byte_trigram'
    , 'codepoint_trigram'
    ]
  fw = OfflineFramework(ALTW2010())

  # This causes the byte tokenstream to be copied into the Store. It is not strictly
  # necessary, but it is useful for if we want to view the original documents via the
  # hydrat browser.
  fw.inducer.process_Dataset(fw.dataset, tss='byte')

  fw.set_class_space('langid')
  fw.set_split('traindev')
  for fs in features:
    fw.set_feature_spaces(fs)
    for l in learners:
      fw.set_learner(l)
      fw.run()
  fw.generate_output()

  # Use the following to upload output to a web server via ssh
  #fw.upload_output('ssh://')

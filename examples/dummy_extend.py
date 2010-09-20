from hydrat.frameworks.preset_split import PresetSplitFramework

import hydrat.corpora.dummy as dummy

import hydrat.classifier.NLTK as nltk
import hydrat.classifier.SVM as svm
import hydrat.classifier.knn as knn
import hydrat.classifier.nearest_prototype as np
import hydrat.classifier.weka as weka
import hydrat.classifier.maxent as maxent

from hydrat.common.transform.weight import TFIDF

if __name__ == "__main__":
  learners=\
    [ knn.cosine_1nnL()
    ]
  ps = PresetSplitFramework(dummy.unicode_dummy())
  ps.set_class_space('dummy_default')
  ps.set_feature_spaces('byte_unigram')
  ps.set_split('dummy_default')
  for l in learners:
    ps.set_learner(l)
    ps.run()
  ps.transform_taskset(TFIDF())
  for l in learners:
    ps.set_learner(l)
    ps.run()
  ps.extend_taskset('codepoint_unigram')
  for l in learners:
    ps.set_learner(l)
    ps.run()
  ps.generate_output()

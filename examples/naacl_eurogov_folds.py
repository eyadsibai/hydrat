import hydrat.corpora.langid.naacl2010 as naacl

import hydrat.classifier.nearest_prototype as np
from hydrat.frameworks.preset_split import PresetSplitFramework

if __name__ == "__main__":
  learners=\
    [ np.cosine_mean_prototypeL()
    , np.skew_mean_prototypeL()
    ]
  ps = PresetSplitFramework(naacl.EuroGOV())
  ps.set_class_space('iso639_1')
  ps.set_feature_space('byte_unigram')
  ps.set_split('crossvalidation')
  for l in learners:
    ps.set_learner(l)
    ps.run()
    ps.generate_output()

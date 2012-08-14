import hydrat.corpora.langid.naacl2010 as naacl
import hydrat.dataset.text as text
import hydrat.classifier.knn as knn
from hydrat.common.transform.featureselect import cavnar_trenkle
from hydrat.frameworks.offline import OfflineFramework 

class EuroGOV(naacl.EuroGOV, text.ByteQuadgram, text.BytePentagram): pass
class TCL(naacl.TCL, text.ByteQuadgram, text.BytePentagram): pass
class Wikipedia(naacl.Wikipedia, text.ByteQuadgram, text.BytePentagram): pass

datasets=\
  [ EuroGOV()
  , TCL()
  , Wikipedia()
  ]

if __name__ == "__main__":
  learner =  knn.oop_1nnL()
  fs = ['byte_unigram', 'byte_bigram', 'byte_trigram', 'byte_quadgram', 'byte_pentagram']
  for ds in datasets:
    fw = OfflineFramework(ds)
    fw.set_class_space('iso639_1')
    fw.set_split('crossvalidation')
    fw.set_feature_spaces(fs)
    
    fw.transform_taskset(cavnar_trenkle)

    fw.set_learner(learner)
    fw.run()

    fw.generate_output()


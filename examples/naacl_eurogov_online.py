from hydrat.corpora.langid import naacl2010
from hydrat.frameworks.online import OnlineFramework
import hydrat.classifier.SVM as svm

if __name__ == "__main__":
  ds = naacl2010.EuroGOV()
  fw = OnlineFramework(ds)
  
  fw.set_feature_spaces('byte_trigram')
  fw.set_class_space('iso639_1')
  fw.set_learner(svm.libsvmExtL(kernel_type='linear'))

  while True:
    try:
      text = raw_input()
    except Exception:
      break
    klass = fw.classify(text)
    print klass

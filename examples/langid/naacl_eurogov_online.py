from hydrat.corpora.langid import naacl2010
from hydrat.frameworks.online import OnlineFramework
import hydrat.classifier.nearest_prototype as np
import hydrat.common.pb as pb

if __name__ == "__main__":
  ds = naacl2010.EuroGOV()
  fw = OnlineFramework(ds)
  
  fw.set_feature_spaces('byte_bigram')
  fw.set_class_space('iso639_1')
  fw.set_learner(np.skew_mean_prototypeL())

  # Silence progressbar
  pb.ENABLED = False
  while True:
    try:
      print ">>>",
      text = raw_input()
    except Exception:
      break
    klass = fw.classify(text)
    print klass[0]

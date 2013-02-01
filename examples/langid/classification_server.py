# Example online classification server
# Marco Lui <saffsd@gmail.com> October 2010
from hydrat.corpora.langid import naacl2010
from hydrat.frameworks.online import OnlineFramework
from hydrat.result.interpreter import SingleHighestValue
import hydrat.classifier.nearest_prototype as np
import hydrat.common.pb as pb

if __name__ == "__main__":
  ds = naacl2010.EuroGOV()
  fw = OnlineFramework(ds)
  
  fw.set_feature_spaces(['byte_unigram','byte_bigram','byte_trigram'])
  fw.set_class_space('iso639_1')
  fw.set_learner(np.skew_mean_prototypeL())
  fw.set_interpreter(SingleHighestValue())

  # Silence progressbar
  pb.ENABLED = False
  fw.serve_xmlrpc()

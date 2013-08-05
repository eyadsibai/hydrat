"""
Interfaces to tools that have not been included into hydrat yet.
"""
import numpy
import os

from hydrat import config
from hydrat.configuration import Configurable, PACKAGE
from hydrat.text import TextClassifier
import hydrat.external.ldig.ldig as ldig

# Detector class taken from shuyo's server.py in the ldig repository
class Detector(object):
    def __init__(self, modeldir):
        self.ldig = ldig.ldig(modeldir)
        self.features = self.ldig.load_features()
        self.trie = self.ldig.load_da()
        self.labels = self.ldig.load_labels()
        self.param = numpy.load(self.ldig.param)

    def detect(self, st):
        label, text, org_text = ldig.normalize_text(st)
        events = self.trie.extract_features(u"\u0001" + text + u"\u0001")
        sum = numpy.zeros(len(self.labels))

        data = []
        for id in sorted(events, key=lambda id:self.features[id][0]):
            phi = self.param[id,]
            sum += phi * events[id]
            data.append({"id":int(id), "feature":self.features[id][0], "phi":["%0.3f" % x for x in phi]})
        exp_w = numpy.exp(sum - sum.max())
        prob = exp_w / exp_w.sum()
        return {"labels":self.labels, "data":data, "prob":["%0.3f" % x for x in prob]}

class LDIG(Configurable, TextClassifier, Detector):
  """
  Shuyo Nakatani's language-identification infinity-grams
  https://github.com/shuyo/ldig
  """
  requires = {
    ('tools', 'ldig-data') : PACKAGE(),
  }

  def __init__(self, model_path=None):
    if model_path is None:
      model_path = os.path.join(config.getpath('tools','ldig-data'), 'model.small')
      model_name = "default"
    else:
      model_name = os.path.basename(model_path)

    self.metadata = dict (
      class_space = 'iso639_1',
      dataset=model_name,
      instance_space='ldig',
      learner='ldig',
      learner_params={},
    )
    TextClassifier.__init__(self)
    Detector.__init__(self, model_path)

  def classify(self, text):
    output = self.detect(text.decode('utf8'))
    return [output['labels'][numpy.argmax(output['prob'])]]

if __name__ == "__main__":
  identifier = LDIG()
  print identifier.classify("This is a test message")
  print identifier.classify("Questa e' la storia di uno di noi")


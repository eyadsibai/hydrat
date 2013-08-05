from hydrat import config
from hydrat.text import TextClassifier
from hydrat.configuration import Configurable, DIR, FILE 

import textcat

class LibTextCat(TextClassifier):
  requires={
    ('tools','libtextcat-config') : FILE('conf.txt'),
    ('tools','libtextcat-base')   : DIR('LM'),
    }

  metadata = dict(
    class_space = 'iso639_1',
    dataset='libtextcat',
    instance_space='libtextcat',
    learner='libtextcat',
    learner_params={},
    )

  def __init__(self):
    TextClassifier.__init__(self, None) # TODO output map
    self.config = config.getpath('tools','libtextcat-config')
    self.base = config.getpath('tools','libtextcat-base')
    self.textcat = textcat.TextCat(self.config, self.base)
    self.metadata['learner_params']['config'] = self.config
    self.metadata['learner_params']['base'] = self.base

  def classify(self, text):
    try:
      return self.textcat.classify(text)
    except (textcat.UnknownException, textcat.ShortException):
      return ['UNKNOWN']

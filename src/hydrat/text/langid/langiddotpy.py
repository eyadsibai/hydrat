from hydrat.text import TextClassifier
from langid.langid import LanguageIdentifier
import os.path

class LangidDotPy(TextClassifier):

  def __init__(self, model_path=None, model_name=None, metadata=None):
    self.model_path = model_path

    if model_name is None:
      if model_path is None:
        model_name = "default"
      else:
        model_name = os.path.basename(model_path)

    self.metadata = dict(
      class_space = 'iso639_1',
      dataset=model_name,
      instance_space='langid.py',
      learner='langid.py',
      learner_params={},
    )
    
    if metadata is not None:
      self.metadata['learner_params'].update(metadata)

    self.identifier = None
    TextClassifier.__init__(self)

  def init_model(self):
    if self.identifier is None:
      if self.model_path is not None:
        try:
          self.identifier = LanguageIdentifier.from_modelpath(self.model_path, norm_probs=False)
        except IOError, e:
          raise ValueError("Failed to unpack model at {0}".format(self.model_path))
      else:
        from langid.langid import model
        self.identifier = LanguageIdentifier.from_modelstring(model, norm_probs=False)

  def classify(self, text):
    self.init_model()
    return [self.identifier.classify(text)[0]]


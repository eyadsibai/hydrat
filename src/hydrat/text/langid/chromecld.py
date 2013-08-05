from hydrat.text import TextClassifier

try:
  import cld
  class ChromeCLD(TextClassifier):
    metadata = dict(
      class_space = 'iso639_1',
      dataset='chromeCLD',
      instance_space='chromeCLD',
      learner='chromeCLD',
      learner_params={},
      )

    def __init__(self):
      TextClassifier.__init__(self, lambda lang: 'UNKNOWN' if lang == 'un' else lang)

    def classify(self, text):
      lang = cld.detect(text)[1]
      return [lang]
except ImportError:
  # cld module not present
  pass

try:
  import cld2
  class ChromeCLD2(TextClassifier):
    metadata = dict(
      class_space = 'iso639_1',
      dataset='chromeCLD2',
      instance_space='chromeCLD2',
      learner='chromeCLD2',
      learner_params={},
      )

    def __init__(self):
      TextClassifier.__init__(self, lambda lang: 'UNKNOWN' if lang == 'un' else lang)

    def classify(self, text):
      try:
        lang = cld2.detect(text)[2][0][1]
      except IndexError:
        lang = 'UNKNOWN'
      return [lang]
except ImportError:
  # cld2 module not present
  pass

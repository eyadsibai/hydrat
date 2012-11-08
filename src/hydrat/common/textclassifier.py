class TextClassifier(object):
  """
  Base class for pre-trained external classifiers
  that accept raw text and produce a prediction.
  """
  def __init__(self, label_map=None):
    """
    @param label_map a function that maps the output of the classifier 
                     onto the final label
    """
    self.label_map = label_map if label_map is not None else lambda x: x

  def classify_batch(self, texts, callback=None):
    retval = []
    for i, t in enumerate(texts):
      retval.append(self.classify(t))
      if callback is not None:
        callback(i)
    return retval

  def __call__(self, text):
    return map(self.label_map,self.classify(text))

import numpy

class Summary(object):
  def __init__(self):
    self.result = None
    self.interpreter = None
    self.handlers = {}

  def init(self, result, interpreter):
    """
    Override this if you need to pre-compute some values if
    a result and/or interpreter change.
    """
    # TODO: How to ensure any extended handlers get a proper init if we override this??
    self.result = result
    self.interpreter = interpreter

  def __call__(self, result, interpreter):
    self.init(result, interpreter)
    return dict( (key, self[key]) for key in self.keys )

  @property
  def local_keys(self):
    for attr in dir(self):
      if attr.startswith('key_'):
        yield(attr.split('_',1)[1])

  @property
  def keys(self):
    for key in self.local_keys:
      yield key
    for key in self.handlers.keys():
      yield key

  def __getitem__(self, key):
    if key in self.local_keys:
      try:
        return getattr(self, 'key_'+key)()
      except KeyError:
        return None
    else: 
      handler = self.handlers[key]
      handler.init(self.result, self.interpreter)
      return handler[key]

  def extend(self, function):
    if hasattr(function, '__class__') and issubclass(function.__class__, Summary):
      # This is an additional summary
      old_keys = set(self.keys)
      new_keys = set(function.keys)
      overlap = old_keys & new_keys
      if len(overlap) != 0:
        raise ValueError, "already have the following keys: " + str(overlap)
      for key in new_keys:
        self.handlers[key] = function
    elif callable(function):
      # This is a wrapped callable
      if function.__name__ in self.keys:
        raise ValueError, "already have the following keys: " + function.__name__
      else:
        setattr(self, 'key_'+ function.__name__, lambda: function(self.result, self.interpreter))
    else:
      raise TypeError, "cannot extend summary with %s" % str(function)


from hydrat.result import CombinedMacroAverage, CombinedMicroAverage
class MicroPRF(Summary):
  def init(self, result, interpreter):
    if result != self.result or interpreter != self.interpreter:
      self.result = result
      self.interpreter = interpreter
      self.micro_score = CombinedMicroAverage(result.overall_confusion_matrix(interpreter))

  def key_micro_precision(self):  return self.micro_score.precision
  def key_micro_recall(self):     return self.micro_score.recall
  def key_micro_fscore(self):     return self.micro_score.fscore

class MacroPRF(Summary):
  def init(self, result, interpreter):
    if result != self.result or interpreter != self.interpreter:
      self.result = result
      self.interpreter = interpreter
      self.macro_score = CombinedMacroAverage(result.overall_confusion_matrix(interpreter))

  def key_macro_precision(self):  return self.macro_score.precision
  def key_macro_recall(self):     return self.macro_score.recall
  def key_macro_fscore(self):     return self.macro_score.fscore

class TimeTaken(Summary):
  def key_avg_learn(self):     return numpy.mean(self.result.individual_metadata('learn_time'))
  def key_avg_classify(self):  return numpy.mean(self.result.individual_metadata('classify_time'))

class Metadata(Summary):
  def __init__(self, keys = []):
    self.__keys = keys
    Summary.__init__(self)

  @property
  def keys(self): return iter(self.__keys)

  def __getitem__(self, key):
    return self.result.metadata.get(key, None)

def classification_summary():
  sf = Summary()
  sf.extend(MacroPRF())
  sf.extend(MicroPRF())
  sf.extend(Metadata(['dataset','class_space','feature_desc','split','learner','learner_params']))
  sf.extend(TimeTaken())
  return sf

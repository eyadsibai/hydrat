import logging
import inspect


def filter_kwargs(fn, skip, kwargs):
  argspec = inspect.getargspec(fn)
  if argspec.keywords is not None:
    supported_kwargs = dict(kwargs)
  else:
    supported_kwargs = dict()
    for key in argspec.args[skip:]:
      try:
        supported_kwargs[key] = kwargs[key]
      except KeyError:
        raise ValueError, "arg %s not available!" % (key)
  return supported_kwargs

class Transformer(object):
  def __init__(self):
    if not hasattr(self, '__name__'):
      self.__name__ = self.__class__.__name__
    self.weights = {}
    self.logger = logging.getLogger(__name__ + '.' + self.__name__)

  def __str__(self):
    return '<Transformer %s>' % self.__name__

  def learn(self, feature_map, class_map):
    raise NotImplementedError

  def apply(self, feature_map):
    raise NotImplementedError

  def _learn(self, feature_map, class_map, add_args):
    supported_kwargs = filter_kwargs(self.learn, 3, add_args)
    retval = self.learn(feature_map, class_map, **supported_kwargs)
    if retval is not None:
      self.logger.critical('learn returned %s', str(retval))

  def _apply(self, feature_map, add_args):
    supported_kwargs = filter_kwargs(self.apply, 2, add_args)
    return self.apply(feature_map, **supported_kwargs)

class LearnlessTransformer(Transformer):
  def learn(self, feature_map, class_map): pass


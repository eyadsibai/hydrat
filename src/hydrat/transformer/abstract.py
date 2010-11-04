import logging

class Transformer(object):
  def __init__(self):
    if not hasattr(self, '__name__'):
      self.__name__ = self.__class__.__name__
    self.weights = {}
    self.logger = logging.getLogger(__name__ + '.' + self.__name__)

  def __str__(self):
    return '<Transformer %s>' % self.__name__

  def learn(feature_map, class_map):
    raise NotImplemented

  def apply(feature_map):
    raise NotImplemented

class LearnlessTransformer(Transformer):
  def learn(self, feature_map, class_map):
    pass


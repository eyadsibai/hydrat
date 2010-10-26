class Transformer(object):
  def __str__(self):
    return '<Transformer %s>' % self.__name__

  def learn(feature_map, class_map):
    raise NotImplemented

  def apply(feature_map):
    raise NotImplemented

class LearnlessTransformer(Transformer):
  def learn(self, feature_map, class_map):
    pass


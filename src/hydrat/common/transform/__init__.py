class Transformer(object):
  def learn(feature_map, class_map):
    raise NotImplemented

  def apply(feature_map):
    raise NotImplemented

class LearnlessTransformer(Transformer):
  def learn(self, feature_map, class_map):
    pass


from copy import deepcopy
class FeatureMap(object):
  def __init__(self, raw, metadata = {}):
    self.raw = raw
    self.metadata = deepcopy(metadata)
    if 'feature_desc' not in metadata:
      self.metadata['feature_desc'] = tuple()

  def __getitem__(self, key):
    # TODO: Take note of what has been selecte somehow?
    return FeatureMap(self.raw[key], self.metadata)

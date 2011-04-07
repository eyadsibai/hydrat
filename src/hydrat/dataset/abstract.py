import logging
import os
from hydrat import config
from hydrat.common.pb import ProgressIter

# TODO: Automatically monkeypatch an instance when a particular ts/fm/cm is loaded, 
#       so we don't try to load it from disk again.
class Dataset(object):
  """ Base class for all datasets. A Dataset is essentially
      a binder for sets of features and classes. The features and classes
      are expressed in a sparse manner, which is usually easier to generate
      from data. hydrat takes care of converting these into dense spases,
      abstracting the data from the labels.
      Deriving classes should implement a set of methods starting with 'cm_'
      and 'fm_' that return class maps and feature maps respectively.
  """
  def __init__(self):
    self.logger = logging.getLogger('hydrat.preprocessor.Dataset')
    # Default fallback for datasets that don't declare a separate name
    if not hasattr(self, '__name__'): 
      self.__name__ = self.__class__.__name__
    # Default fallback for datasets that don't declare an instance space
    if not hasattr(self, 'instance_space'):
      self.instance_space = self.__name__
    # We need to avoid circularity in working out instance ids
    self._checked_dep = set()
    self._instance_ids = None

  def __str__(self):
    ret_strs = []
    ret_strs.append("Name : %s"% self.__name__ )
    ret_strs.append("Size : %d instances" % len(self.instance_ids))
    ret_strs.append("Features: ")
    for f in self.featuremap_names:
      ret_strs.append("  %s" % f)
    ret_strs.append("Classes: ")
    for c in self.classmap_names:
      ret_strs.append("  %s" % c)
    splits = list(self.split_names)
    if len(splits) > 0:
      ret_strs.append("Splits: ")
      for s in splits:
        ret_strs.append("  %s" % s)
    tokenstreams = list(self.tokenstream_names)
    if len(tokenstreams) > 0:
      ret_strs.append("TokenStreams: ")
      for t in tokenstreams:
        ret_strs.append("  %s" % t)
    return '\n'.join(ret_strs)

  def classspace(self, name):
    return getattr(self, 'cs_'+name)()

  def classmap(self, name):
    return getattr(self, 'cm_'+name)()

  def featuremap(self, name):
    return getattr(self, 'fm_'+name)()

  def split(self, name):
    return getattr(self, 'sp_'+name)()

  def tokenstream(self, name):
    return getattr(self, 'ts_'+name)()

  def sequence(self, name):
    return getattr(self, 'sq_'+name)()

  def prefixed_names(self, prefix):
    for key in dir(self):
      if key.startswith(prefix + '_'):
        yield key.split('_',1)[1] 
    
  @property 
  def classmap_names(self): 
    return self.prefixed_names('cm')

  @property 
  def featuremap_names(self): 
    return self.prefixed_names('fm')

  @property 
  def classspace_names(self): 
    return self.prefixed_names('cs')

  @property 
  def split_names(self): 
    return self.prefixed_names('sp')

  @property 
  def tokenstream_names(self): 
    return self.prefixed_names('ts')

  @property 
  def sequence_names(self): 
    return self.prefixed_names('sq')

  def identifiers(self):
    # Check with the class maps first as they are usually 
    # smaller and thus quicker to load. 
    # Then try fm and ts in that order.
    prefixes = ['cm', 'fm', 'ts']
    for p in prefixes:
      for name in self.prefixed_names(p):
        key = '_'.join((p, name))
        if key not in self._checked_dep:
          self._checked_dep.add(key)
          method = getattr(self, key)
          ids = method().keys()
    raise NotImplementedError, "No tokenstreams, feature maps or class maps defined!"

  @property
  def instance_ids(self):
    if self._instance_ids is None:
      self._instance_ids = list(sorted(self.identifiers()))
    self.logger.debug("instance_ids %s %s", self.__name__, str(self._checked_dep))
    return self._instance_ids



  def features(self, tsname, extractor):
    """
    Generate feature map by applying an extractor to a
    named tokenstream.
    """
    tokenstream = self.tokenstream(tsname)
    # TODO: Instead of a dict, use a disk-backed data structure. Would be useful if there was some way
    # to avoid duplicating keys perhaps.
    fm = {}

    for instance_id in ProgressIter(tokenstream, label="Processing Documents"):
      fm[instance_id] = extractor(tokenstream[instance_id])
      if len(fm[instance_id]) == 0:
        msg =  "%s_%s has no tokens for '%s'" % (tsname, extractor.__name__, instance_id)
        if config.getboolean('debug','allow_empty_instance'):
          self.logger.warning(msg)
        else:
          raise ValueError, msg

    return fm


import logging
import os
from hydrat.common.pb import ProgressIter

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

  @property
  def instance_ids(self):
    # Check with the class maps first as they are usually 
    # smaller and thus quicker to load
    try: 
      names = self.classmap_names
      ids = set(self.classmap(names.next()).keys())
    except StopIteration:
      try:
        names = self.featuremap_names
        ids = set(self.featuremap(names.next()).keys())
      except StopIteration:
        raise NotImplementedError, "No feature maps or class maps defined!"
    return list(sorted(ids))

  def features(self, tsname, extractor):
    """
    Generate feature map by applying an extractor to a
    named tokenstream.
    """
    tokenstream = self.tokenstream(tsname)
    fm = {}

    for instance_id in ProgressIter(tokenstream, label="Processing Documents"):
      fm[instance_id] = extractor(tokenstream[instance_id])
      if len(fm[instance_id]) == 0:
        self.logger.warning( "TokenStream '%s' has no tokens for '%s'", tsname, instance_id )

    return fm

class SingleDir(Dataset):
  """ Mixin for a dataset that has all of its source text files
  in a single directory. Requires that the deriving class
  implements a data_path method.
  """
  def data_path(self):
    raise NotImplementedError, "Deriving class must implement this"

  def text(self):
    path = self.data_path()
    instances = {}
    for filename in os.listdir(path):
      filepath = os.path.join(path, filename)
      if os.path.isfile(filepath):
        instances[filename] = open(filepath).read()
    return instances


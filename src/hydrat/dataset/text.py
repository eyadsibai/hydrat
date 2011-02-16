import os
from hydrat.dataset import Dataset
import hydrat.common.extractors as ext

class TextDataset(Dataset):
  """ Base class for datasets where instances can be represented
      as single string. Ideal for traditional text classification
      tasks.

      The only requirement for subclassing TextDataset is that the 
      subclass must implement the ts_byte method, which returns a 
      dictionary mapping from the instance identifier to the
      text of the instance.
  """
  def ts_byte(self):
    """
    Return a dictionary from instance identifiers
    to the content of the instance in a string 
    This should be a normal byte string.
    """
    raise NotImplementedError

class SingleDir(TextDataset):
  """ Mixin for a dataset that has all of its source text files
  in a single directory. Requires that the deriving class
  implements a data_path method.
  """
  def data_path(self):
    raise NotImplementedError, "Deriving class must implement this"

  def ts_byte(self):
    path = self.data_path()
    instances = {}
    for filename in os.listdir(path):
      filepath = os.path.join(path, filename)
      if os.path.isfile(filepath):
        instances[filename] = open(filepath).read()
    return instances

class FilePerClass(TextDataset):
  def data_path(self):
    raise NotImplementedError, "Deriving class must implement this"

  def ts_byte(self):
    path = self.data_path()
    ts = {}
    for cl in os.listdir(path):
      with open(os.path.join(path, cl)) as f:
        for i,instance in enumerate(f):
          instance_id = '%s%d'%(cl, i)
          ts[instance_id] = instance
    return ts

  def cm_filename(self):
    path = self.data_path()
    cm = {}
    for cl in os.listdir(path):
      with open(os.path.join(path, cl)) as f:
        for i,instance in enumerate(f):
          instance_id = '%s%d'%(cl, i)
          cm[instance_id] = [cl]
    return cm

class ByteUnigram(TextDataset):
  def fm_byte_unigram(self):   return self.features('byte', ext.unigram)

class ByteBigram(TextDataset):
  def fm_byte_bigram(self):    return self.features('byte', ext.bigram)

class ByteTrigram(TextDataset):
  def fm_byte_trigram(self):   return self.features('byte', ext.trigram)

class ByteQuadgram(TextDataset):
  def fm_byte_quadgram(self):  return self.features('byte', ext.quadgram)

class BytePentagram(TextDataset):
  def fm_byte_pentagram(self): return self.features('byte', ext.pentagram)

class ByteUBT(ByteUnigram, ByteBigram, ByteTrigram): pass

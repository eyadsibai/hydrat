import os
import csv

from collections import defaultdict

from hydrat import config
from hydrat.dataset.iso639 import ISO639_1
from hydrat.dataset.text import ByteUBT
from hydrat.dataset.encoded import CodepointUBT, UTF8


from hydrat.dataset import Dataset

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

class NAACL2010(ISO639_1, SingleDir):
  """ Mixin for NAACL2010 dataset, which has a standardized format
  for the metadata file.
  """
  rawdata_path = config.get('corpora', 'naacl2010-langid')

  def data_path(self): return os.path.join(self.rawdata_path, self.__name__)
  def meta_path(self): return self.data_path+'.meta'

  def cm_iso639_1(self):
    cm = {}
    with open(self.meta_path(), 'r') as meta:
      reader = csv.reader(meta)
      for row in reader:
        docid, encoding, lang, partition = row
        cm[docid] = lang
    return cm
  
class EuroGOV(NAACL2010, UTF8, ByteUBT, CodepointUBT):
  __name__ = 'EuroGOV'

class TCL(NAACL2010, ByteUBT, CodepointUBT):
  __name__ = 'TCL'

  def encodings(self):
    encodings = {}
    with open(self.meta_path(), 'r') as meta:
      reader = csv.reader(meta)
      for row in reader:
        docid, encoding, lang, partition = row
        encodings[docid] = encoding
    return encodings
    
class Wikipedia(NAACL2010, UTF8, ByteUBT, CodepointUBT):
  __name__ = 'Wikipedia'

import os
from hydrat import config
from hydrat.dataset.text import ByteUBT
from hydrat.dataset.encoded import CodepointUBT
from collections import defaultdict
import xml.etree.ElementTree as e

class UDHR(ByteUBT, CodepointUBT):
  """Backend for UDHR data"""
  __name__ = "UDHR"
  __data = None
  path = config.getpath('corpora','UDHR')
  __index = None
  __docids = None

  def __init__(self):
    ByteUBT.__init__(self)
    CodepointUBT.__init__(self)

  @property
  def _index(self):
    if self.__index is None:
      r = {}
      index = e.parse(os.path.join(self.path,'index.xml')).getroot()
      for entry in index:
        a = entry.attrib
        id = 'udhr_' + a['l'] + ( '_' + a['v'] if 'v' in a else '')
        classes = {}
        for key in ['uli', 'bcp47', 'ohchr', 'country', 'region', 'l']:
          if a[key].strip():
            classes[key] = a[key].strip()
        r[id] = classes
      self.__index = r
    return self.__index
  
  @property
  def _docids(self):
    if self.__docids is None:
      self.__docids = [os.path.splitext(f)[0] for f in os.listdir(self.path) if f.endswith('.txt')]
    return self.__docids

  def encodings(self):
    return defaultdict(lambda:'utf-8')

  def ts_byte(self):
    data = {}
    for id in self._docids:
      f = open(os.path.join(self.path, id+'.txt'))
      data[id] = '\n'.join( l for l in f.readlines()[5:])
    return data

  def index_classmap(self, param):
    r = {};  i = self._index
    for id in self._docids:
      r[id] = [i[id][param]] if param in i[id] else []
    return r

  def cm_uli(self): return self.index_classmap('uli')
  def cm_bcp47(self): return self.index_classmap('bcp47')
  def cm_ohchr(self): return self.index_classmap('ohchr')
  def cm_ethnologue(self): return self.index_classmap('l')

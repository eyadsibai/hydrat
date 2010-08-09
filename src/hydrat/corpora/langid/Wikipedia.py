from __future__ import with_statement
import os
from hydrat.preprocessor.dataset.text import ByteUBTQP
from hydrat.preprocessor.dataset.encoded import CodepointUBT, UTF8
from hydrat import configuration
from collections import defaultdict
from cPickle import load

from iso639 import ISO639_1

class WikipediaPickledDataset(ByteUBTQP, CodepointUBT, UTF8, ISO639_1):
  wikipath = configuration.get('corpora','wikipedia')
  __data   = None

  def __init__(self, name, filename):
    ByteUBTQP.__init__(self)
    CodepointUBT.__init__(self)
    self.__name__ = name
    self.filename = filename 

  @property
  def _data(self):
    if self.__data is None:
      with open(os.path.join(self.wikipath, self.filename)) as pickle:
        self.__data = load(pickle)
    return self.__data

  def __repr__(self):
    return "<WikipediaPickledDataset of %s>" % self.filename

  def text(self):
    instances = self._data[0]
    return instances

  def cm_wikipedia_prefix(self):
    cm = self._data[1]
    return cm

  def cm_iso639_1(self):
    cm = self.cm_wikipedia_prefix()
    for id in cm:
      # The 2-letter wikipedia prefixes are all valid ISO639 labels
      # TODO: Check that they actually correspond!
      cm[id] = [ label if len(label) == 2 else 'UNKNOWN' for label in cm[id] ]
    return cm
          
def FiveK():
  return WikipediaPickledDataset('wikipedia_fivek', 'fivek.pickle')

def FiveK_2seg():
  return WikipediaPickledDataset('wikipedia_fivek_2seg', 'fivek_2seg.pickle')

def SubOneK():
  raise NotImplementedError, "Need to implement subset on new-type datasets"
  backend = FiveK()

if __name__ == "__main__":
  x = FiveK()
  y = FiveK_2seg()
  print dir(x)
    

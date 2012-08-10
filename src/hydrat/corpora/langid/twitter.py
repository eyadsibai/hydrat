"""
hydrat dataset interface to datasets based on twitter data
"""
import csv
from hydrat import config
from hydrat.dataset.text import ByteUBT, DirPerClass
from hydrat.dataset.encoded import CodepointUBT, UTF8
from hydrat.dataset.iso639 import ISO639_1
from hydrat.configuration import Configurable, DIR

class TwitterZHENJA5k(UTF8, ISO639_1, Configurable, ByteUBT, CodepointUBT):
  """Backend for Twitter zh-en-ja task"""
  requires={
    ('corpora','twitter-zhenja-5k') : DIR('twitter-zhenja-5k'),
    }
  datapath = config.getpath('corpora', 'twitter-zhenja-5k')

  def cm_iso639_1(self):
    retval = {}
    with open(self.datapath) as f:
      for row in f:
        row = row.split('\t')
        retval[row[0]] = [ row[1] ]
    return retval

  def ts_byte(self):
    retval = {}
    with open(self.datapath) as f:
      for row in f:
        row = row.split('\t')
        retval[row[0]] = row[2] 
    return retval

class TwitterSCarter(UTF8, ISO639_1, DirPerClass, Configurable, ByteUBT, CodepointUBT):
  requires={ ('corpora', 'scarter-twitter') : DIR('scarter-twitter') }

  def data_path(self):
    return config.getpath('corpora', 'scarter-twitter')

  def cm_iso639_1(self):
    return self.classmap('dirname')

class TwitterBenelearn11(UTF8, ISO639_1, DirPerClass, Configurable, ByteUBT, CodepointUBT):
  requires={ ('corpora', 'benelearn11-twitter') : DIR('benelearn11-twitter') }

  def data_path(self):
    return config.getpath('corpora', 'benelearn11-twitter')

  def cm_iso639_1(self):
    return self.classmap('dirname')

if __name__ == "__main__":
  x = TwitterZHENJA5k()
  print x
  import pdb;pdb.set_trace()

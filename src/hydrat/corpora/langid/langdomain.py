"""
Cross-domain langid corpus
"""
import os
join = os.path.join

from hydrat.dataset.text import ByteUBT
from hydrat.proxy import DataProxy
from hydrat.configuration import Configurable, DIR
from hydrat.dataset.iso639 import ISO639_1
from hydrat import config


class LangidDomain(Configurable, ISO639_1, ByteUBT):
  """
  Cross-domain langid corpus based on langdomain-v1, see readme for details:

  ..todo: Manage encodings
  """
  requires={ ('corpora', 'langdomain-v1') : DIR('langdomain-v1') }
  
  def identifiers(self):
    dp = config.getpath('corpora', 'langdomain-v1')
    with open(join(dp, 'paths')) as pathlist:
      for line in pathlist:
        line = line.strip()
        if not line: continue
        line, instance = os.path.split(line)
        line, lang = os.path.split(line)
        line, domain = os.path.split(line)
        yield '-'.join((domain, lang, instance))
    
  def ts_byte(self):
    dp = config.getpath('corpora', 'langdomain-v1')
    ts = {}
    for i in self.instance_ids:
      domain, lang, instance = i.split('-',2)
      ts[i] = open(join(dp, domain, lang, instance)).read()
    return ts
      
  def cm_iso639_1(self):
    cm = {}
    for i in self.instance_ids:
      domain, lang, instance = i.split('-',2)
      cm[i] = [lang]
    return cm

  def cm_domain(self):
    cm = {}
    for i in self.instance_ids:
      domain, lang, instance = i.split('-',2)
      cm[i] = [domain]
    return cm

if __name__ == "__main__":
  ds = LangidDomain()
  print ds


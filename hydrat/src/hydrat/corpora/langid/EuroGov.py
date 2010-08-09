from __future__ import with_statement
import os
from hydrat.preprocessor.dataset.text import ByteUBTQP
from hydrat.preprocessor.dataset.encoded import BagOfWords, CodepointUBT, UTF8
from hydrat import configuration
from collections import defaultdict

from iso639 import ISO639_1
class EuroGov(BagOfWords, ByteUBTQP, CodepointUBT, UTF8, ISO639_1):
  __name__ = 'EuroGov'
  eurogovpath = configuration.getpath('dotwp1')
  segments = {"trn":"training", "tst":"test", "dev":"development"}

  def text(self):
    instances = {}
    for prefix in self.segments:
      path = os.path.join(self.eurogovpath, self.segments[prefix])
      for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):
          instance_id = prefix + filename
          instances[instance_id] = open(filepath).read()
    return instances

  def eurogov_classes(self, label):
    cm = {}
    for prefix in self.segments:
      # All this work is required in order to determine the full
      # set of instance identifiers
      path = os.path.join(self.eurogovpath, self.segments[prefix])
      for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):
          instance_id = unicode(prefix + filename)
          cm[instance_id] = []

      # Read in the actual class labels
      cl_file = self.segments[prefix] + '-' + label
      cl_path = os.path.join(self.eurogovpath, cl_file)
      with open(cl_path) as f:
        for line in f:
          filename, classlabel = line.split()
          instance_id = unicode(prefix + filename)
          cm[instance_id].append(classlabel)
    return cm

  def cm_eurogov_prefix(self):
    return self.eurogov_classes('lang')

  def cm_iso639_1(self):
    return self.cm_eurogov_prefix()
          
  def cm_eurogov_docclass(self):
    return self.eurogov_classes('docclass')
          
if __name__ == "__main__":
  x = EuroGov()
  print dir(x)
    

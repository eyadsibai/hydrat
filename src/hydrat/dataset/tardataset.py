"""
Hydrat corpus interface for dir-per-class corpora stored
in tar format.

Marco Lui, January 2013
"""

import tarfile
import os
from hydrat.dataset.text import DirPerClass

class TarDataset(DirPerClass):
  #def __init__(self, *args, **kwargs):
  #  DirPerClass.__init__(self, *args, **kwargs)

  def identifiers(self):
    path = self.data_path()
    with tarfile.open(path) as f:
      ids = [ m.name for m in f if m.isfile() ]
    return ids

  def ts_byte(self):
    path = self.data_path()
    with tarfile.open(path) as f:
      # We read all the files into memory at one go. This may be costly
      # if the archive is big, but random access to compressed archives
      # is likely to be even worse.
      ts = dict( (m.name, f.extractfile(m).read()) for m in f if m.isfile() )
    return ts

  def cm_dirname(self):
    path = self.data_path()
    with tarfile.open(path) as f:
      # We read all the files into memory at one go. This may be costly
      # if the archive is big, but random access to compressed archives
      # is likely to be even worse.
      cm = dict( (m.name, [os.path.dirname(m.name)]) for m in f if m.isfile() )
    return cm 


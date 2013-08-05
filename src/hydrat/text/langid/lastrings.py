from hydrat.text import TextClassifier
from hydrat import config
from hydrat.configuration import Configurable, EXE, FILE

import subprocess
import tempfile
import os
import csv
from contextlib import closing

class WhatLang(Configurable, TextClassifier):
  """
  Wrapper for the whatlang identifier embedded in la-strings
  http://sourceforge.net/projects/la-strings/
  """
  requires={
    ('tools','whatlang')        : EXE('whatlang'),
    ('tools','whatlang-db')        : FILE('languages.db'),
    }

  metadata = dict(
    class_space = 'iso639_1',
    dataset='la-strings',
    instance_space='whatlang',
    learner='whatlang',
    learner_params={},
    )

  def __init__(self):
    self.toolpath  = config.getpath('tools','whatlang')
    self.tooldb    = config.getpath('tools','whatlang-db')
    self.tempdir   = config.getpath('paths','scratch')
    TextClassifier.__init__(self, lambda l: l if len(l) == 2 else 'UNKNOWN')

  def classify(self, text):
    with tempfile.NamedTemporaryFile(delete=False, dir=self.tempdir) as f:
      f.write(text)
    cmd = [self.toolpath,'-l'+self.tooldb,'-b0','-n1','-t',f.name]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out = p.stdout.read()
    lang = out.split(':')[0]

    os.unlink(f.name)
    retval =  [ lang ]
    return retval

  def classify_batch(self, texts, callback=None):
    cmd = [self.toolpath,'-l'+self.tooldb,'-b1','-n1','-t']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate('\n'.join(l.replace('\n',' ') for l in texts))
    retval = [ [row.split('\t',1)[0]] for row in stdout.splitlines() ]
    return retval


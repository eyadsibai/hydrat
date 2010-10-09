"""
Wrapper for the famous TextCat
http://www.let.rug.nl/~vannoord/TextCat/text_cat.tgz
"""
import tempfile
import os
import shutil
import itertools
import operator

from subprocess import Popen, PIPE, STDOUT

from hydrat import config
from hydrat.configuration import Configurable, EXE
from hydrat.common.pb import ProgressIter


class TextCat(Configurable):
  requires =\
    { ('tools', 'textcat')    : EXE('text_cat')
    }

  def __init__(self):
    self.model_path = None
    self.toolpath = config.getpath('tools','textcat')

  def __del__(self):
    if self.model_path is not None:
      shutil.rmtree(self.model_path)

  def train(self, pairs):
    # Create a temporary directory to store models
    self.model_path = tempfile.mkdtemp(prefix='textcat', dir=config.getpath('paths','scratch'))

    key = operator.itemgetter(1)
    for klass, group in itertools.groupby(sorted(pairs,key=key),key):
      class_path = os.path.join(self.model_path, klass + '.lm')
      class_data = '\n'.join(t for t,c in group)
      p = Popen([self.toolpath, '-n'], stdout=open(class_path,'w'), stdin=PIPE, stderr=STDOUT)
      p.communicate(input=class_data)[0]
    

  def classify(self, text):
    p = Popen\
          ( [self.toolpath, '-u1', '-d%s'%self.model_path]
          , stdout=PIPE
          , stdin=PIPE
          , stderr=STDOUT
          )
    out = p.communicate(input=text)[0]
    return out.strip()

  def batch_classify(self, texts):
    return map(self.classify, texts)


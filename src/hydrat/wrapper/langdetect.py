"""
Wrapper for language-detection
http://code.google.com/p/language-detection/
"""
import tempfile
import os
import shutil
import itertools
import operator

from subprocess import Popen, PIPE, STDOUT

class LangDetect(object):
  def __init__(self, javapath, toolpath, profilespath, scratch):
    self.profilespath = profilespath
    self.javapath = javapath
    self.toolpath = toolpath
    self.scratch = scratch

  def train(self, pairs):
    # Training is supported by the package, but the package also ships with premade profiles
    # so we use it as an off-the-shelf reference
    raise NotImplementedError

  def classify(self, text):
    # java -jar lib/langdetect.jar --detectlang -d [profile directory] [test file(s)]
    testfile = tempfile.NamedTemporaryFile(dir=self.scratch, prefix='LangDetect-')
    testfile.write(text)
    testfile.flush()
    p = Popen\
          ( [self.javapath, '-jar', self.toolpath, '--detectlang', '-d', self.profilespath, testfile.name ]
          , stdout=PIPE
          , stdin=None
          , stderr=STDOUT
          )
    out = p.communicate()[0]
    if p.returncode == 0:
      return out.split(':')[1][1:]
    raise ValueError, "Error in underlying library"

  def batch_classify(self, texts):
    return map(self.classify, texts)


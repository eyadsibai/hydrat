"""
Wrapper for blacklist-classifier
https://bitbucket.org/tiedemann/blacklist-classifier/wiki/Home

Marco Lui, January 2013

USAGE
   Classification:
          blacklist_classifier [OPTIONS] lang1 lang2 ... < file

   training:
          blacklist_classifier -n [OPTIONS] text1 text2 > blacklist.txt
          blacklist_classifier [OPTIONS] -t "t1.txt t2.txt ..." lang1 lang2 ...

   run experiments:
          blacklist_classifier -t "t1.txt t2.txt ..." \
                                  -e "e1.txt e2.txt ..." \
                                  lang1 lang2 ...

   command line arguments:
        lang1 lang2 ... are language ID's
        blacklists are expected in <BlackListDir>/<lang1-lang2.txt
        t1.txt t2.txt ... are training data files (in UTF-8)
        e1.txt e2.txt ... are training data files (in UTF-8)
        the order of languages needs to be the same for training data, eval data
          as given by the command line arguments (lang1 lang2 ..)


        -a <freq> ...... min freq for common words
        -b <freq> ...... max freq for uncommon words
        -c <score> ..... min difference score to be relevant
        -d <dir> ....... directory of black lists
        -i ............. classify each line separately
        -m <number> .... use approximately <number> tokens to train/classify
        -n ............. train a new black list
        -v ............. verbose mode

        -U ............. don't lowercase
        -S ............. don't tokenize (use the string as it is)
        -A ............. don't discard tokens with non-alphabetic characters

"""
import os, tempfile, shutil
from subprocess import Popen, PIPE, STDOUT

class BlacklistClassifier(object):
  def __init__(self, toolpath, langs):
    self.toolpath = toolpath
    self.langs = list(langs)

  def train(self, pairs):
    # Training is supported by the package but we have not yet implemented
    # the management thereof.
    raise NotImplementedError

  def classify(self, text):
    p = Popen(\
          [ self.toolpath ] + self.langs,
          stdin=PIPE,
          stdout=PIPE,
          stderr=PIPE,
          )
    out = p.communicate(text)
    return out

if __name__ == "__main__":
  tool = '/usr/local/bin/blacklist_classifier'
  c = BlacklistClassifier(tool, '~/TEMP', ['bs','hr','sr'])

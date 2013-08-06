"""
Interface for Martin Majlis' YALI.

https://github.com/martin-majlis/YALI

An invocation looks like:
../detector-perl/detect.pl --ngram=4 --dict=model.limited-800 --lang-file=languages-list/languages-google.txt --freq-norm=1

And stdin is interpreted as a list of files.
"""
import os,sys
import tempfile
import subprocess
from collections import defaultdict
from itertools import islice

from hydrat.text import TextClassifier
from hydrat import config
from hydrat.configuration import Configurable, EXE, FILE


class YALI(Configurable, TextClassifier):
  requires={
    ('tools','yali')                : EXE('detect.pl'),
    ('tools','yali-dict')           : FILE('model.limited-800'),
    ('tools','yali-langfile')       : FILE('languages-google.txt'),
    ('tools','yali-outmap')         : FILE('languages-google-map'),
    }

  def __init__(self,ngram=4, freq_norm=1, batchsize=800):
    self.ngram = ngram
    self.freq_norm = freq_norm
    self.batchsize = batchsize
    self.toolpath  = config.getpath('tools','yali')
    self.tooldict  = config.getpath('tools','yali-dict')
    self.toollang  = config.getpath('tools','yali-langfile')
    self.tempdir   = config.getpath('paths','scratch')

    outmap = defaultdict(lambda: 'UNKNOWN')
    with open(config.getpath('tools','yali-outmap')) as f:
      for line in f:
        k, v = line.split()
        outmap[k] = v

    TextClassifier.__init__(self, outmap.get)
    self.metadata = dict(
      class_space = 'iso639_1',
      dataset='yali',
      instance_space='yali',
      learner='yali',
      learner_params=dict(ngram=ngram, freq_norm=freq_norm),
    )

  def classify(self, text):
    with tempfile.NamedTemporaryFile(delete=False, dir=self.tempdir) as f:
      f.write(text)

    cmd = [
      self.toolpath,
      '--ngram={}'.format(self.ngram),
      '--freq-norm={}'.format(self.freq_norm),
      '--dict={}'.format(self.tooldict),
      '--lang-file={}'.format(self.toollang),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate(f.name)
    lang = stdout.split()[1]

    os.unlink(f.name)
    retval =  [ lang ]
    return retval

  def classify_batch(self, texts, callback=None):
    t_iter = iter(texts)
    processed = 0

    cmd = [
      self.toolpath,
      '--ngram={}'.format(self.ngram),
      '--freq-norm={}'.format(self.freq_norm),
      '--dict={}'.format(self.tooldict),
      '--lang-file={}'.format(self.toollang),
    ]

    retval = []
    while True:
      batch = islice(t_iter, self.batchsize)
      files = []
      for t in batch:
        with tempfile.NamedTemporaryFile(delete=False, dir=self.tempdir) as f:
          files.append(f.name)
          f.write(t)

      p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      stdout, stderr = p.communicate('\n'.join(files))

      for l in stdout.splitlines():
        try:
          filename, label = l.split()
          retval.append([self.label_map(label)])
        except ValueError:
          retval.append([])
        processed += 1

      for f in files:
        os.unlink(f)

      callback(processed)
      if processed == len(texts):
        # Done processing all texts
        break

    return retval

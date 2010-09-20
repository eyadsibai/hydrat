"""
Python wrapper for GibbsLDA++
http://gibbslda.sourceforge.net/
"""
import subprocess
import tempfile
import os
import shutil
import pexpect
import re
import numpy
import scipy.sparse
import logging

logger = logging.getLogger(__name__)


LDA_EXE = ''
TMP = './scratch'

RE_ITERATION = re.compile(r'Iteration (?P<count>\d+) ...')

def array2file(array, fileobj):
  fileobj.write(str(array.shape[0]) + '\n')
  for row in array:
    instance = []
    for feature in row.nonzero()[1]:
      for i in xrange(int(row[0,feature])):
        instance.append(str(feature))
    fileobj.write(' '.join(instance) + '\n')

class GibbsLDA(object):
  def __init__( self
              , alpha = None
              , beta = 0.1
              , ntopics = 100
              , niters = 2000
              , infiters = 30
              , savestep = None
              , twords = 0
              , exe = LDA_EXE
              , tmp = TMP
              , clear_temp = True
              ):
    if alpha is None:
      self.alpha = 50 / ntopics
    else:
      self.alpha = alpha
    self.beta = beta
    self.ntopics = ntopics
    self.niters = niters
    self.infiters = infiters
    self.savestep = savestep if savestep is not None else 0
    self.twords = twords
    self.clear_temp = clear_temp
    self.exe = exe
    self.tmp = tmp
    self.workdir = os.path.abspath(tempfile.mkdtemp(prefix='GibbsLDA',dir=self.tmp))
    self.trained = False
    if not (os.path.exists(exe) and os.access(exe, os.X_OK)):
      raise ValueError, "'%s' is not a valid executable" % exe
 
  def __del__(self):
    if self.clear_temp and os.path.exists(self.workdir):
      shutil.rmtree(self.workdir)

  def estimate(self, feature_map, progress_callback = None):
    """
    <model_name>.phi: 
      This file contains the word-topic distributions, i.e., p(wordw|topict). 
      Each line is a topic, each column is a word in the vocabulary
    <model_name>.theta: 
      This file contains the topic-document distributions, i.e., p(topict|documentm). 
      Each line is a document and each column is a topic.
    <model_name>.tassign: 
      This file contains the topic assignment for words in training data. 
      Each line is a document that consists of a list of <wordij>:<topic of wordij>
      Could use this as a token stream!
    """
    with tempfile.NamedTemporaryFile\
            ( prefix='GibbsLDA-'
            , suffix='-learn'
            , dir=self.workdir
            , delete=self.clear_temp
            ) as f:
      # Write the training file
      array2file(feature_map, f)
      command =\
        [ self.exe
        , '-est'
        , '-alpha', self.alpha
        , '-beta', self.beta
        , '-ntopics', self.ntopics
        , '-niters', self.niters
        , '-savestep', self.savestep
        , '-twords', self.twords
        , '-dfile', f.name
        ]
      command = ' '.join(map(str, command))
      logger.debug(command)
      lda_instance = pexpect.spawn(command)
      lda_instance.expect(r'Sampling (?P<count>\d+) iterations!')
      niters = int(lda_instance.match.group('count'))

      for i in range(niters):
        lda_instance.expect(RE_ITERATION)
        if progress_callback is not None:
          progress_callback(i+1)

      lda_instance.expect(r'Saving the final model!')
      lda_instance.expect(pexpect.EOF)
    self.trained = True

  @property
  def topics(self):
    if not self.Trained:
      raise ValueError, "Not trained"
    theta = numpy.genfromtxt(os.path.join(self.workdir,'model-final.theta'))
    theta = scipy.sparse.csr_matrix(theta)
    return theta

  def continue_estimate(self):
    raise NotImplementedError, "Continued estimation not yet implemented"
    

  def apply(self, feature_map, progress_callback=None):
    with tempfile.NamedTemporaryFile\
            ( prefix='GibbsLDA-'
            , suffix='-apply'
            , dir=self.workdir
            , delete=self.clear_temp
            ) as f:
      array2file(feature_map, f)
      command =\
        [ self.exe
        , '-inf'
        , '-dir', self.workdir
        , '-model', 'model-final'
        , '-niters', self.infiters
        , '-twords', self.twords
        , '-dfile', os.path.basename(f.name)
        ]
      command = ' '.join(map(str, command))
      logger.debug(command)

      lda_instance = pexpect.spawn(command)
      lda_instance.expect(r'Sampling (?P<count>\d+) iterations for inference!')
      niters = int(lda_instance.match.group('count'))

      for i in range(niters):
        lda_instance.expect(RE_ITERATION)
        if progress_callback is not None:
          progress_callback(i+1)

      lda_instance.expect(r'Saving the inference outputs!')
      lda_instance.expect(pexpect.EOF)
      theta = numpy.genfromtxt(f.name+'.theta')
    theta = scipy.sparse.csr_matrix(theta)
    return theta

    

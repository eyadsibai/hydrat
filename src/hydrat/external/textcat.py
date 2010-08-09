import logging
import tempfile
import os
import shutil
import numpy
import datetime as dt
import time
from subprocess import Popen, PIPE, STDOUT
from hydrat.preprocessor.model.inducer import class_matrix 
from hydrat.common import progress
from hydrat import config
from hydrat.external import select


try:
  toolpath = config.get('tools','textcat')
  class TextCat(object):
    __name__ = "textcat_external"
    logger = logging.getLogger('hydrat.external.TextCat')
    toolpath = config.get('tools','textcat')

    def __init__(self):
      self.model_path = None
      self.num_classes = None

    def __del__(self):
      if self.model_path is not None:
        shutil.rmtree(self.model_path)

    def train(self, instances, classmap):
      start_time = time.time()
      self.logger.info("Training")
      if self.model_path is not None:
        self.logger.critical('Already been trained!')
        raise NotImplementedError

      # Create a temporary directory to store models
      self.model_path = tempfile.mkdtemp()

      self.num_classes = classmap.shape[1]
      for i in range(self.num_classes):
        self.logger.info("Training Class %d of %d", i+1, self.num_classes)
        class_name = 'class%d' % i
        class_indices = classmap[:,i]
        class_data = '\n'.join(select(instances, class_indices))
        class_path = os.path.join(self.model_path, class_name + '.lm')
        p = Popen([self.toolpath, '-n'], stdout=open(class_path,'w'), stdin=PIPE, stderr=STDOUT)
        out = p.communicate(input=class_data)[0]
      
      self.logger.info("Models for %d classes at %s", self.num_classes, self.model_path)
      self.train_time = time.time() - start_time

    def classify(self, instances):
      start_time = time.time()
      result = numpy.zeros((len(instances), self.num_classes), dtype='bool')
      
      def report(i,t): 
        self.logger.info('Processing entry %d of %d (%d%% done)', i+1, t, i*100/t)

      for i,instance in enumerate(progress(instances,10,report)):
        p = Popen([self.toolpath, '-u1', '-d%s'%self.model_path], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        out = p.communicate(input=instance)[0]
        cl_ind = int(out.strip()[5:])
        result[i,cl_ind] = True
      self.classify_time = time.time() - start_time
      return result

except KeyError:
  pass

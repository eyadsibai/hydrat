from hydrat.classifier.abstract import Learner, Classifier
from itertools import izip
from hydrat import config
from hydrat.configuration import Configurable, EXE

import tempfile
import os
import logging
import time
import numpy
import sys


class CRFFileWriter(object):

  @staticmethod
  def instance(fv, cv = None):
    fv = fv.toarray()
    
    if cv is not None:
      classids = numpy.arange(cv.shape[0])[cv]
    else:
      classids = [""] 
    classlabelblock = ",".join(str(id) for id in classids)
    features = " ".join(    str(fv[0,i]) 
                       for  i
                       in   xrange(len(fv[0])) 
                       )
    return "%s %s\n" % (features, classlabelblock)

  @staticmethod
  def writefile(file, fvs, sequence, cvs = None):
    # Identify all the nodes without parents - these are the starts of sequences
    first_post = numpy.nonzero(sequence.sum(0)==0)[0]
    first_post_index = first_post.tolist()[0]
    if cvs is not None: assert fvs.shape[0] == cvs.shape[0]
    # For each of the start nodes
    for i in first_post_index:
      # Follow the chain of nodes, writing features as we go
      while(len(sequence[i].nonzero()[1])>0):
        if cvs is not None:
          one_line = CRFFileWriter.instance(fvs[i], cvs[i])
        else:
          one_line = CRFFileWriter.instance(fvs[i])
        i = sequence[i].nonzero()[1][0]
        file.write(one_line)
      # Write out the final instance
      if cvs is not None:
        one_line = CRFFileWriter.instance(fvs[i], cvs[i])
      else:
        one_line = CRFFileWriter.instance(fvs[i])
      file.write(one_line)
      file.write("\n")



class crfsgdL(Configurable, Learner):
  __name__ = 'crfsgd'
  requires =\
    { ('tools','crfsgd')            : EXE('crfsgd')
    , ('tools','crfsgd-conlleval')  : EXE('conlleval')
    }

  def __init__(self, capacity=1.0):
    self.clear_temp = config.getboolean('debug', 'clear_temp_files')
    self.toolpath = config.getpath('tools', 'crfsgd')
    self.capacity = capacity
    Learner.__init__(self)

    self.model_path = None

  def __del__(self):
    if self.clear_temp:
      if self.model_path is not None: os.remove(self.model_path)

  def _params(self):
    return dict(capacity=self.capacity)
    
  def _learn(self, feature_map, class_map, sequence):
    writer = CRFFileWriter
    
    #build the template for CRF learner
    #TODO: note that I did not put *identifiers* at the moment 
    template_len = feature_map.shape[1]
    template = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    template.write("# Unigram\n")
    self.logger.debug("writing template file: %s", template.name)
    for i in range(template_len):
      template.write("U"+str(i)+":%x[0,"+str(i)+"]\n")
    template.flush()

     #Create and write the training file
    train = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    self.logger.debug("writing training file: %s", train.name)
    writer.writefile(train, feature_map, sequence, class_map)
    train.flush()

    #Create a temporary file for the model
    model_file, self.model_path = tempfile.mkstemp()
    self.logger.debug("model path: %s", self.model_path)
    os.close(model_file)

    training_command =\
      "%s -e %s -q -c %f %s %s %s" % ( self.toolpath 
                      , config.getpath('tools', 'crfsgd-conlleval')
                      , self.capacity
                      , self.model_path
                      , template.name
                      , train.name
                      )
    self.logger.debug("Training CRF: %s", training_command)
    # Try to replace os.popen with subprocess (http://docs.python.org/library/subprocess.html)
    # Alternatively, if there is progress output from 'crfsgd', use Pexpect (http://www.noah.org/wiki/Pexpect)
    process = os.popen(training_command)
    output = process.read()
    return_value = process.close()
    if return_value:
      self.logger.critical("Training 'crfsgd' failed with output:")
      self.logger.critical(output)
      raise ValueError, "Training 'crfsgd' returned %s"%(str(return_value))
    return crfsgdC( self.model_path
                  , self.toolpath
                  , class_map.shape[1]
                  )

class crfsgdC(Classifier):
  __name__ = "crf"

  def __init__(self, model_path, toolpath, num_classes):
    Classifier.__init__(self)
    self.sequence = None
    self.model_path  = model_path
    self.toolpath  = toolpath 
    self.num_classes = num_classes
    self.clear_temp  = config.getboolean('debug','clear_temp_files')

  def __invoke_classifier(self, test_path):
    #Create a temporary file for the results
    result_file, result_path = tempfile.mkstemp()
    os.close(result_file)
    classif_command = "%s -t %s %s %s" % ( self.toolpath 
                                      , self.model_path
                                      , test_path
                                      , ">"+result_path
                                      )
    self.logger.debug("Classifying CRF: %s", classif_command)
    process = os.popen(classif_command)
    output = process.read()
    return_value = process.close()
    if return_value:
      self.logger.critical("Classifying 'crfsgd' failed with output:\n"+output)
      raise ValueError, "Classif 'crfsgd' returned %s"%(str(return_value))
    return result_path

  def __parse_result(self, result_path, num_test_docs):
    first_post = numpy.nonzero(self.sequence.sum(0)==0)[0]
    first_post_index = first_post.tolist()[0]
    child_sequence = self.sequence

    result_file = open(result_path)
    result_lines = result_file.readlines()
    classifications = numpy.zeros((num_test_docs, self.num_classes), dtype='bool')

    j = 0
    checksum = 0
    for i in first_post_index:
      while(len(child_sequence[i].nonzero()[1])>0):
        class_index = int(result_lines[j].split()[-1])
        classifications[i, class_index] = True
        checksum += 1
        j += 1
        i = child_sequence[i].nonzero()[1][0]
      class_index = int(result_lines[j].split()[-1])
      classifications[i, class_index] = True
      checksum += 1
      assert result_lines[j+1] == "\n"
      j += 2
    assert checksum == num_test_docs
    # Dispose of the unneeded output file
    result_file.close()
    if self.clear_temp:
      os.remove(result_path)

    return classifications

  def write_testfile(self, test, feature_map):
    writer = CRFFileWriter

     #Create and write the test file
    self.logger.debug("writing test file: %s", test.name)
    writer.writefile(test, feature_map, self.sequence)
    test.flush()

  def _classify(self, feature_map, sequence):
    self.sequence = sequence
    test  = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    self.write_testfile(test, feature_map)
    num_test_docs = feature_map.shape[0]

    return self.classify_from_file(test.name, num_test_docs)

  def classify_from_file(self, test_path, num_test_docs):
    result_path = self.__invoke_classifier(test_path)
    classifications = self.__parse_result(result_path, num_test_docs)
    return classifications



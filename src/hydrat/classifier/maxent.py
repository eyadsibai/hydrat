from hydrat.classifier.abstract import Learner, Classifier
from hydrat import config

import tempfile
import os
import numpy

from hydrat.classifier.SVM import SVMFileWriter
from hydrat.configuration import is_exe

"""
Wrapper for Le Zhang's maxent toolkit
http://homepages.inf.ed.ac.uk/lzhang10/maxent_toolkit.html
"""

class maxentLearner(Learner):
  __name__ = "maxent"
  requires =\
    { 'maxent' : 'maxent' 
    }

  def __init__(self, iterations=3, method='lbfgs' ):
    self.toolpath = config.get('tools','maxent')
    Learner.__init__(self)
    if method not in ['lbfgs','gis']:
      raise ValueError, "Invalid method '%s'"%method
    self.iterations = iterations
    self.method = method
    self.model_path = None
    self.clear_temp = True 

  def _check_installed(self):
    if not is_exe(self.toolpath):
      raise ValueError, "Tool not installed!"

  def _params(self):
    return dict( iterations = self.iterations
              , method = self.method
              )

  def _learn(self, feature_map, class_map):
    writer = SVMFileWriter

    #Create and write the training file
    train = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    self.logger.debug("writing training file: %s", train.name)
    writer.writefile(train, feature_map, class_map)
    train.flush()

    #Create a temporary file for the model
    model_file, self.model_path = tempfile.mkstemp()
    self.logger.debug("model path: %s", self.model_path)
    os.close(model_file)

    train_path = train.name 
    training_command = "%s %s -b -m %s -i %d" % ( self.toolpath 
                                                , train_path
                                                , self.model_path 
                                                , self.iterations
                                                )
    self.logger.debug("Training maxent: %s", training_command)
    process = os.popen(training_command)
    output = process.read()
    return_value = process.close()
    if return_value:
      self.logger.critical("Training maxent failed with output:")
      self.logger.critical(output)
      raise ValueError, "Training maxent returned %s"%(str(return_value))

    return maxentClassifier( self.model_path
                          , class_map.shape[1]
                          , self.__name__
                          )

  def __del__(self):
    if self.clear_temp:
      if self.model_path is not None: os.remove(self.model_path)

class maxentClassifier(Classifier):
  __name__ = "maxent"

  def __init__(self, model_path, num_classes, name=None):
    self.toolpath = config.get('tools','maxent')
    if name:
      self.__name__ = name
    Classifier.__init__(self)
    self.model_path  = model_path
    self.num_classes = num_classes
    self.clear_temp  = True# Clear temp files after execution
  
  def __invoke_classifier(self, test_path):
    #Create a temporary file for the results
    result_file, result_path = tempfile.mkstemp()
    os.close(result_file)

    classif_command = "%s -p -m %s --detail -o %s %s" % ( self.toolpath 
                                                        , self.model_path
                                                        , result_path
                                                        , test_path
                                                        )
    self.logger.debug("Classifying maxent: %s", classif_command)
    process = os.popen(classif_command)
    output = process.read()
    return_value = process.close()
    if return_value:
      self.logger.critical("Classifying maxent failed with output:\n"+output)
      raise ValueError, "Classif maxent returned %s"%(str(return_value))

    return result_path 

  def __parse_result(self, result_path, num_test_docs):
    result_file = open(result_path)
    classifications = numpy.zeros((num_test_docs, self.num_classes), dtype='float')

    for i,line in enumerate(result_file):
      terms = line.split()
      while terms != []:
        # Read pairs of outcome, probability
        outcome = int(terms.pop(0))
        probability = float(terms.pop(0))
        classifications[i, outcome] = probability

    # Dispose of the unneeded output file
    result_file.close()
    if self.clear_temp:
      os.remove(result_path)

    return classifications


  def write_testfile(self, test, feature_map):
    writer = SVMFileWriter

    #Create and write the test file
    self.logger.debug("writing test file: %s", test.name)
    writer.writefile(test, feature_map)
    test.flush()

  def _classify(self, feature_map):
    test  = tempfile.NamedTemporaryFile(delete=self.clear_temp)
    self.write_testfile(test, feature_map)
    num_test_docs = feature_map.shape[0]

    return self.classify_from_file(test.name, num_test_docs)

  def classify_from_file(self, test_path, num_test_docs):
    result_path = self.__invoke_classifier(test_path)
    classifications = self.__parse_result(result_path, num_test_docs)
    return classifications

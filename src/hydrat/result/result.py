import logging

import numpy

from hydrat.common.invert_dict import invert_dict
from hydrat.common.richcomp import RichComparisonMixin
from hydrat.result import confusion_matrix 

def result_from_task(task, classifications, metadata = {}):
  """
  Build a result object by combining a task and the classifications
  resulting from running the task, along with any additional 
  metadata to be saved.
  """
  goldstandard     = task.test_classes
  classifications  = classifications
  instance_indices = numpy.flatnonzero(task.test_indices)
  full_metadata    =  {}
  full_metadata.update(task.metadata)
  full_metadata.update(metadata)

  return Result(goldstandard, classifications, instance_indices, full_metadata)

class Result(RichComparisonMixin):
  """
  Encapsulates the output of a classification, together with 
  enough data to meaningfully be able to interpret it

  The metadata will be important for deep interpretation.
  For example:
    'classifier' stores the name of the classifier used.
                 This is needed to decide how to interpret the 
                 classification output (eg lower-better or higher-better)
    'dataset'    stores the name of the underlying dataset.
                 This is essential in recovering information
                 about document and class labels.
                 The 'goldstandard' results are stored for convenience,
                 but should actually be identical to those of the dataset,
                 caveat being that there will only be a subset of the dataset
                 present in the classification.
                 TODO: Work out how this is being managed!! Are we able
                 to fully recover the docid/classlabel for a set of results?

  @param goldstandard: The gold standard classification outputs. 
                       In cases where this is unknown this may just
                       be an empty or a random boolean array.
                       axis 0: instance
                       axis 1: class
  @type goldstandard: 2-d boolean array
  @param classifications: The raw outputs of the classifier. 
                          Will often be a floating-point value.
                          Examples of lowest-best: distance metrics
                          Examples of highest-best: pseudo-probabilities from naive bayes
                          axis 0: instance
                          axis 1: class
  @type classifications: 2-d array
  """
  def __init__( self
              , goldstandard
              , classifications
              , instance_indices
              , metadata = {} 
              ):
    self.logger = logging.getLogger("hydrat.result.Result")
    self.goldstandard     = goldstandard
    self.classifications  = classifications
    self.instance_indices = instance_indices
    assert goldstandard.dtype     == numpy.bool
    assert self.goldstandard.shape == self.classifications.shape
    assert len(instance_indices) == len(self.goldstandard)

    self.metadata = {}
    self.metadata.update(metadata)

  def __repr__(self):
    return "<result " + str(self.metadata) + ">"

  def __str__(self):
    output = ["<Result>"]
    output.append("    %15s : %s" % ('classif_size', str(self.classifications.shape)))
    for key in self.metadata:
      output.append("    %15s : %s" % (str(key), str(self.metadata[key])))
    return '\n'.join(output)

  def __eq__(self, other):
    """
    Define two results as identical iff they have the same metadata
    and the same classifier outputs
    """
    try:
      return self._eq_metadata(other) and self._eq_data(other)
    except AttributeError:
      return False

  def _eq_metadata(self, other):
    """
    Test for equality over the metadata
    """
    metadata_conditons = [    self.metadata[c] == other.metadata[c]
                        for  c
                        in   [ "class_uuid"
                             , "dataset_uuid"
                             , "classifier"
                             , "feature_desc"
                             , "task_type"
                             ]
                        ]
    return all(metadata_conditons)

  def _eq_data(self, other):
    """
    Test for equality over the data itself
    """
    # Check that there are the same number of instances in each result
    # Then check that they are the same instances
    if len(self.instance_indices) != len(other.instance_indices) \
      or (self.instance_indices != other.instance_indices).any():
      return False
    conditions = [ (self.goldstandard     == other.goldstandard).all()
                 , (self.classifications  == other.classifications).all()
                 ]
    return all(conditions)

  def classification_matrix(self, interpreter):
    """
    @param interpreter: How to interpret the classifier output
                        for purposes of constructing the matrix
    @type interpreter: ResultInterpreter instance
    @return: an array of classification counts
             axis 0 is Goldstandard
             axis 1 is Classifier Output 
    """
    classifications = interpreter(self.classifications)
    doc_count, class_count = classifications.shape
    matrix = numpy.empty((class_count, class_count), dtype='int64')
    for gs_i in xrange(class_count):
      for cl_i in xrange(class_count):
        gs = self.goldstandard[:,gs_i]
        cl = classifications[:,cl_i]
        matrix[gs_i,cl_i] = numpy.logical_and(gs,cl).sum()
    return matrix

  def confusion_matrix(self, interpreter):
    """
    @param interpreter: How to interpret the classifier output
                        for purposes of constructing the matrix
    @type interpreter: ResultInterpreter instance
    @return: A perclass confusion matrix, with classes stacked in the
             order they are classified in
    """
    classifications = interpreter(self.classifications)
    return confusion_matrix(self.goldstandard, classifications)
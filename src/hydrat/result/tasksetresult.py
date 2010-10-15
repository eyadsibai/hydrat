import numpy
from hydrat.result import fscore
from hydrat.common.richcomp import RichComparisonMixin
from collections import defaultdict

class TaskSetResult(RichComparisonMixin):
  #Contains a raw result and descriptive frills
  # TODO: Add descriptive frills
  def __init__(self, raw_results, metadata = None):
    self.raw_results = raw_results
    if metadata:
      # Copy the metadata if some is provided
      self.metadata = dict(metadata)
    else:
      self.metadata = {}

  def individual_metadata(self, raw_result_key):
    return [ r.metadata[raw_result_key] for r in self.raw_results ]
    
  def __repr__(self):
    return "<TaskSetResult %s>"% (str(self.metadata))

  def __str__(self):
    output = ["<TaskSetResult>"]
    for key in self.metadata:
      output.append("    %15s : %s" % (str(key), str(self.metadata[key])))
    return '\n'.join(output)

  def __eq__(self, other):
    try:
      cond_1 = len(self.raw_results) == len(other.raw_results)
      cond_2 = all( r in self.raw_results for r in other.raw_results )
      return cond_1 and cond_2
    except AttributeError:
      return False

  ###
  # Should work on providing access to taskset-level
  # confusion and classification matrices.
  ###
  def overall_classification(self, indices): 
    r = self.raw_results[0]
    num_inst = len(indices)
    num_class = r.goldstandard.shape[1]
    num_res = len(self.raw_results)
    result = numpy.zeros((num_inst, num_class, num_res), dtype=r.classifications.dtype)

    for r_i, r in enumerate(self.raw_results): 
      r_map = dict( (k,v) for v,k in enumerate(r.instance_indices))
      for i in indices: 
        result[i,:,r_i] = r.classifications[r_map[i]]
    return result
      

      
    

  def overall_classpairs(self, interpreter):
    """
    Return a mapping (from, to) -> [indices] extended over all folds
    """
    mapping = defaultdict(list)
    for r in self.raw_results:
      cl = r.classpairs(interpreter)
      for key in cl:
        mapping[key].extend(cl[key])
    return mapping

  def overall_classification_matrix(self, interpreter):
    """
    Sums the classification matrix of each result
    @return: axis 0 - Goldstandard
             axis 1 - Classifier Output
    @rtype: 2-d array
    """
    return sum([ r.classification_matrix(interpreter) for r in self.raw_results ])

  def overall_confusion_matrix(self, interpreter):
    """
    Provides all confusion matrices in a form easy to compute scores for
    @return: axis 0 - results 
             axis 1 - classes
             axis 2 - tp, tn, fp, fn
    @rtype: 3-d array
    """
    return numpy.array([ r.confusion_matrix(interpreter) for r in self.raw_results ])
    

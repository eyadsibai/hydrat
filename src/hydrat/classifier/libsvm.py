"""
SVM classifier using libsvm's python API
Results are unexpectedly different from the CLI version (and worse!)
"""
from hydrat.classifier.abstract import Learner, Classifier
from hydrat.classifier.common import sparse2sparse_dict
import svm

__all__ = \
  [ "libsvmL"
  , "libsvmProbL" 
  ]

class libsvmLearner(Learner):
  def __init__(self, param, name):
    self.param = param
    if param.probability == 1:
      self.__name__ = 'libsvmProb_' + str(name) 
    else:
      self.__name__ = 'libsvm_' + str(name)
    Learner.__init__(self)

  def _learn(self, feature_map, class_map):
    self.logger.debug("Converting labels")
    self.logger.debug("Converting samples")

    samples = []; labels = []
    for c,r in izip(class_map, feature_map):
      #import pdb;pdb.set_trace()
      sd = sparse2sparse_dict(r)
      for class_index in numpy.flatnonzero(c):
        labels.append(int(class_index))
        samples.append(sd)
        
    # libsvm API supports lists for dense representation and dicts for sparse representation
    # Minimal experimentation suggests that sparse dicts are much faster. Degree of
    # sparseness probably plays a factor, have not investigated further.
    # TODO: This class labelling is wrong! This forces a one-of-m task!
    # raise NotImplemented
    #labels = [ int(r.argmax()) for r in class_map]
    #samples = [ dict( (i,int(s)) for i,s in enumerate(r) if s != 0) for r in feature_map ]
    #samples = [ sparse2sparse_dict(r) for r in feature_map ]

    self.logger.debug("Formulating problem")
    problem = svm.svm_problem(labels, samples)
    self.logger.debug("Learning Model")

    # Override stderr to avoid seeing libsvm output
    sys.stderr.flush()
    err = open('/dev/null', 'a+', 0)
    orig = os.dup(sys.stderr.fileno())
    os.dup2(err.fileno(), sys.stderr.fileno())

    # learn model
    model = svm.svm_model(problem, self.param)

    # Undo the override
    os.dup2(orig, sys.stderr.fileno())
    os.close(orig)

    if self.param.probability == 1:
      return libsvmProbClassifier(model, self.__name__)
    else:
      return libsvmClassifier(model, self.__name__)

class libsvmClassifier(Classifier): 
  def __init__(self, model, name):
    self.__name__ = name
    self.model = model
    Classifier.__init__(self)

  def _classify(self, feature_map):
    out = numpy.zeros((feature_map.shape[0], self.model.get_nr_class()), dtype=bool)
    for i,r in enumerate(feature_map):
      sample = sparse2sparse_dict(r) 
      label = self.model.predict(sample)
      out[i, label] = True
    return out

class libsvmProbClassifier(libsvmClassifier):
  def _classify(self, feature_map):
    out = numpy.zeros((feature_map.shape[0], self.model.get_nr_class()), dtype=float)
    for i,r in enumerate(feature_map):
      sample = sparse2sparse_dict(r)
      label, values = self.model.predict_probability(sample)
      for j in values:
        out[i,j] = values[j]
    return out

def libsvmL():
  param = svm.svm_parameter()
  return libsvmLearner(param, 'default')

def libsvmProbL():
  param = svm.svm_parameter(kernel_type = svm.RBF, C=1, probability = 1)
  return libsvmLearner(param, 'default')


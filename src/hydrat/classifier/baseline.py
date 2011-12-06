from abstract import Learner, Classifier
import random
import numpy

__all__=['randomL','majorityL']


class majorityL(Learner):
  __name__ = 'majority'

  def __init__(self, n=1):
    Learner.__init__(self)
    self.n = n
  
  def _check_installed(self):
    pass

  def _params(self):
    return dict(n=self.n)

  def _learn(self, feature_map, class_map):
    return majorityC(class_map, self.n)


class majorityC(Classifier):
  """ Implements a simple majority-class classifier """
  __name__ = "majorityclass"

  def __init__(self, class_map, n):
    Classifier.__init__(self)
    self.n = n
    self.class_map = class_map

  def _classify(self, test_fm):
    num_docs         = test_fm.shape[0]
    num_classes      = self.class_map.shape[1]
    frequencies      = self.class_map.sum(0)
    majority_classes = frequencies.argsort()[-self.n:]
    classifications  = numpy.zeros((num_docs, num_classes), dtype='bool')
    classifications[:,majority_classes] = True
    return classifications

from hydrat.common.sampling import CheckRNG
class randomL(Learner):
  __name__ = 'random'
  
  @CheckRNG
  def __init__(self, rng=None):
    Learner.__init__(self)
    self.rng = rng

  def _check_installed(self):
    pass

  def _params(self):
    return dict(rng_state = rng.get_state())

  def _learn(self, feature_map, class_map):
    return randomC(feature_map, class_map, self.rng)


class randomC(Classifier):
  """ Random classifier- classifies documents at random.
      Respects the training distribution by sampling classifications
      from it
  """
  __name__ = "randomclass"
  def __init__(self, feature_map, class_map, rng):
    Classifier.__init__(self)
    self.fm = feature_map
    self.cm = class_map
    self.rng = rng

  def _classify(self, test_fm):
    train_docs = self.fm.shape[0]
    test_docs  = test_fm.shape[0]
    test_doc_indices = self.rng.sample(xrange(train_docs), test_docs)
    return self.cm[test_doc_indices]

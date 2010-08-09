from abstract import Learner, Classifier
import random
import numpy

__all__=['randomL','majorityL']


class majorityL(Learner):
  __name__ = 'majority'

  def _params(self):
    return dict()

  def _learn(self, feature_map, class_map):
    return majorityC(class_map)


class majorityC(Classifier):
  """ Implements a simple majority-class classifier """
  __name__ = "majorityclass"

  def __init__(self, class_map):
    Classifier.__init__(self)
    self.class_map = class_map

  def _classify(self, test_fm):
    num_docs         = test_fm.shape[0]
    num_classes      = self.class_map.shape[1]
    frequencies      = self.class_map.sum(0)
    majority_class   = frequencies.argmax() #could sort instead for top N classes
    classifications  = numpy.zeros((num_docs, num_classes), dtype='bool')
    classifications[:,majority_class] = True
    return classifications

class randomL(Learner):
  __name__ = 'random'
  
  def __init__(self, seed=None):
    Learner.__init__(self)
    self.seed = seed

  def _params(self):
    return dict(seed=self.seed)

  def _learn(self, feature_map, class_map):
    return randomC(feature_map, class_map, self.seed)


class randomC(Classifier):
  """ Random classifier- classifies documents at random.
      Respects the training distribution by sampling classifications
      from it
  """
  __name__ = "randomclass"
  def __init__(self, feature_map, class_map, seed):
    Classifier.__init__(self)
    self.fm = feature_map
    self.cm = class_map
    self.rng = random.Random()
    if seed != None:
      self.rng.seed(seed)

  def _classify(self, test_fm):
    train_docs = self.fm.shape[0]
    test_docs  = test_fm.shape[0]
    test_doc_indices = self.rng.sample(xrange(train_docs), test_docs)
    return self.cm[test_doc_indices]

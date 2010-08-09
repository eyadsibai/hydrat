"""
A Task is a container for a pairing of training data and test data.
It is useful in the context of techniques such as cross validation, 
where the same set of training and test data must be reused in 
different experiments.
Basic tasks are just little boxes that store training and test data.
Slightly more sophisticated tasks will contain a reference to a data
store and will obtain data on demand.
The most sophisiticated tasks will involve multiple data stores, as
well as a method for reconciling the data.
"""
import numpy

class Task(object):
  __slots__ = [ 'train_vectors'
              , 'train_classes'
              , 'test_vectors'
              , 'test_classes'
              , 'metadata'
              , 'train_indices'
              , 'test_indices'
              ]

class InMemoryTask(Task):
  """Task where the feature map and class map are entirely in-memory"""
  __slots__ = Task.__slots__ + [ 'class_map', 'feature_map' ]
  def __init__(self, feature_map, class_map, train_indices, test_indices, metadata):
    self.class_map = class_map
    self.feature_map = feature_map
    self.train_indices = train_indices
    self.test_indices = test_indices
    self.metadata = dict(metadata)
    
  @property
  def train_vectors(self):
    """
    Get training instances
    @return: axis 0 is instances, axis 1 is features
    @rtype: 2-d array
    """
    if not issubclass(self.train_indices.dtype.type, numpy.bool_):
      raise ValueError, 'Expected a boolean selection map'

    return self.feature_map[self.train_indices.nonzero()[0]]

  @property
  def test_vectors(self):
    """
    Get test instances
    @return: axis 0 is instances, axis 1 is features
    @rtype: 2-d array
    """
    if not issubclass(self.test_indices.dtype.type, numpy.bool_):
      raise ValueError, 'Expected a boolean selection map'

    return self.feature_map[self.test_indices.nonzero()[0]]

  @property
  def train_classes(self):
    """
    Get train classes 
    @return: axis 0 is instances, axis 1 is classes 
    @rtype: 2-d array
    """
    return self.class_map[self.train_indices]

  @property
  def test_classes(self):
    """
    Get test classes 
    @return: axis 0 is instances, axis 1 is classes 
    @rtype: 2-d array
    """
    return self.class_map[self.test_indices]

"""
=====================
:mod:`hydrat.dataset`
=====================
This module provides support classes for making it easy to define classes that meet 
hydrat's dataset interface.
"""

import logging

class Dataset(object):
  """ Base class for all datasets. A Dataset is essentially
      a binder for sets of features and classes. The features and classes
      are expressed in a sparse manner, which is usually easier to generate
      from data. hydrat takes care of converting these into dense spases,
      abstracting the data from the labels.
      Deriving classes should implement a set of methods starting with 'cm_'
      and 'fm_' that return class maps and feature maps respectively.
  """

  def __init__(self):
    self.logger = logging.getLogger('hydrat.preprocessor.Dataset')

  def classspace(self, name):
    return getattr(self, 'cs_'+name)()

  def classmap(self, name):
    return getattr(self, 'cm_'+name)()

  def featuremap(self, name):
    return getattr(self, 'fm_'+name)()

  def prefixed_names(self, prefix):
    for key in dir(self):
      if key.startswith(prefix + '_'):
        yield key.split('_',1)[1] 
    
  @property 
  def classmap_names(self): 
    return self.prefixed_names('cm')

  @property 
  def featuremap_names(self): 
    return self.prefixed_names('fm')

  @property 
  def classspace_names(self): 
    return self.prefixed_names('cs')

  @property
  def instance_ids(self):
    # Check with the class maps first as they are usually 
    # smaller and thus quicker to load
    try: 
      names = self.classmap_names
      ids = set(self.classmap(names.next()).keys())
    except StopIteration:
      try:
        names = self.featuremap_names
        ids = set(self.featuremap(names.next()).keys())
      except StopIteration:
        raise NotImplementedError, "No feature maps or class maps defined!"
    return list(sorted(ids))

def check_dataset(ds):
  """ Perform a check on a dataset object to ensure it has been implemented correctly. 
  Will raise an exception if something is wrong.
  """
  #TODO IMPLEMENT THIS
  pass

import configobj
import os

class HydratConfiguration(object):
  """
  hydrat configuration is handled by means of the configobj
  third-party library.

  In its current implementation, hydrat's configuration simply
  consists of an abstraction of a set of paths. Querying the 
  configuration object for a label will return the associated path
  (if set). This could be expanded to other options if needed.

  A convenience is provided by allowing paths to be specified
  relative to other paths. This avoids duplication of paths, 
  and also makes the relationship between paths clearer.
  """
  def __init__(self, path):
    assert(os.path.exists(path))
    self.config = configobj.ConfigObj(path)

  def _expand_path(self,path):
    """
    Recursive mechanism to expand path tuples into a full path
    """
    if isinstance(path, list):
      head = path.pop(0)
      head_path = self._expand_path(self.config['paths'][head][:])
      tail_path = reduce(os.path.join, path)
      return os.path.join(head_path, tail_path)
    else: 
      return path

  def getpath(self, label):
    """
    Look up a label in the configuration
    """
    path = self.config['paths'][label][:]
    if isinstance(path, list):
      return self._expand_path(path)
    else:
      return path

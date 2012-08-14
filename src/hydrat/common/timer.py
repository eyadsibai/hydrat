import time

class Timer(object):
  """
  Simple timing context manager. Sets an attribute "duration"
  which records the last interval timed.
  """
  def __init__(self):
    self.duration = None
    pass

  def __enter__(self):
    self.start = time.time()

  def __exit__(self, ty, val, tb):
    self.duration = time.time() - self.start


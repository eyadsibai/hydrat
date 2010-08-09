import logging

class Suppress(logging.Filter):
  def filter(self, record):
    return not logging.Filter.filter(self,record)

def getHydraFormatter():
  return logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def getHydraLogger(name = None):
  logger = logging.getLogger("hydrat"+ ('.'+name if name is not None else ''))
  logger.setLevel(logging.DEBUG)
  formatter = getHydraFormatter() 

  # Set up logging to stdout
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  ch.setFormatter(formatter)

  # Suppress svmfilewriter output
  filter = Suppress('hydrat.classifier.svmfilewriter')
  ch.addFilter(filter)

  logger.addHandler(ch) 

  return logger

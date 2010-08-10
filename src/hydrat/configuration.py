import ConfigParser
import logging
import os.path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = '.hydratrc'
#TODO:
# Need to build the default configuration by scanning all the tools and corpora
# for their individual configuration requirements by some means of introspection
# Also, check each tool for possible installed locations automagically
# Need to expand user's home path automatically
# Allow configurable logging level

def default_configuration():
  """
  Default configuration options
  """
  default_config = ConfigParser.SafeConfigParser()

  default_config.add_section('paths')
  default_config.set('paths', 'work', './work')
  default_config.set('paths', 'scratch', '/tmp')
  default_config.set('paths', 'models', '%(work)s/models')
  default_config.set('paths', 'tasks', '%(work)s/tasks')
  default_config.set('paths', 'results', '%(work)s/results')
  default_config.set('paths', 'output', '%(work)s/output')

  default_config.add_section('tools')
  default_config.set('tools', 'bin', '/usr/bin')
  default_config.set('tools', 'rainbow', '%(bin)s/rainbow')
  default_config.set('tools', 'libsvm', '%(bin)s')
  default_config.set('tools', 'bsvm', '%(bin)s')
  default_config.set('tools', 'java', '%(bin)s/java')
  default_config.set('tools', 'weka', '%(bin)s/weka.jar')
  default_config.set('tools', 'maxent', '%(bin)s')
  default_config.set('tools', 'textcat', '%(bin)s')
  default_config.set('tools', 'libs', '%(bin)s')

  default_config.add_section('corpora')
  default_config.set('corpora', 'corpora', '~/data')
  default_config.set('corpora', 'reuters', '%(corpora)s/reut21578')
  default_config.set('corpora', 'eurogov', '%(corpora)s/eurogov')
  default_config.set('corpora', 'tcl', '%(corpora)s/TCL')
  default_config.set('corpora', 'udhr', '%(corpora)s/udhr/txt')
  default_config.set('corpora', 'wikipedia', '%(corpora)s/wikipedia')

  default_config.add_section('logging')
  default_config.set('logging', 'level', 'debug')
  default_config.set('logging', 'format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  return default_config

def write_default_configuration(file=DEFAULT_CONFIG_FILE):
  """ Write the default configuration to a file
  @param file File name or file-like object
  """
  if isinstance(file, str):
    file = open(file, 'w')
  config = default_configuration()
  config.write(file)

def read_configuration(additional_path=[]):
  """ Read a configuration file
  @param additional_path Additional path(s) to consider
  """
  if isinstance(additional_path, str):
    additional_path = [additional_path]
  all_paths = [os.path.expanduser('~/'+DEFAULT_CONFIG_FILE),'./'+DEFAULT_CONFIG_FILE ]
  all_paths.extend(additional_path)

  config = default_configuration()
  paths = config.read(all_paths)
  logger.debug("Read configuration from %s", str(paths))
  return config

logger = logging.getLogger("hydrat")
logger.setLevel(logging.DEBUG)
console_output = logging.StreamHandler()
console_output.setLevel(logging.CRITICAL)
logger.addHandler(console_output)

LEVELS =\
  { 'debug': logging.DEBUG
  , 'info': logging.INFO
  , 'warning': logging.WARNING
  , 'error': logging.ERROR
  , 'critical': logging.CRITICAL
  }

def process_configuration(config):
  """ Process the hydrat configuration file. Should be invoked whenever the configuration changes.
      Right now this is mostly only useful for setting up logging.
  """
  # Process options related to logging
  global console_output 
  level = LEVELS.get(config.get('logging','level'), logging.NOTSET)
  formatter = logging.Formatter(config.get('logging','format',raw=True))
  console_output.setLevel(level)
  console_output.setFormatter(formatter)
   


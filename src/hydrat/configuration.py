import ConfigParser
import logging
import os
import numpy

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = '.hydratrc'
#TODO:
# Build the default configuration by scanning all the tools and corpora
#   for their individual configuration requirements by some means of introspection
# Check each tool for possible installed locations automagically
#   - Implemented, set up for svms. Still needs work.
#   - Will not work for the 'external' package, for libs and svm, since they don't inherit
#     learner. Might want to have a 'configurable' mixin
# Configure java-based packages better. Can base on java-wrappers from debian
# Move back to configObj maybe. Problems with ConfigParser:
#   1) Outputs key-values in any order in a section
#   2) Does not allow you to inset comments

class HydratConfigParser(ConfigParser.SafeConfigParser):
  """ HydratConfigParser adds a getpath method, to postprocess paths in a 
  config file by running expanduser and expandvars on the path.
  """
  def getpath(self, section, option):
    "Do some post-processing on a received path"
    path = self.get(section, option)
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    return path

def all_subclasses(klass):
  subclasses = klass.__subclasses__()
  result = set(subclasses)
  for k in subclasses:
    result.update(all_subclasses(k))
  return result

def is_exe(fpath):
  return os.path.exists(fpath) and os.access(fpath, os.X_OK)

def which(program):
  # from http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
  fpath, fname = os.path.split(program)
  if fpath:
    if is_exe(program):
      return program
  else:
    for path in os.environ["PATH"].split(os.pathsep):
      exe_file = os.path.join(path, program)
      if is_exe(exe_file):
        return exe_file

  return None

def default_configuration():
  """
  Default configuration options
  """
  default_config = HydratConfigParser()

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
  default_config.set('tools', 'java', '%(bin)s/java')
  default_config.set('tools', 'weka', '%(bin)s/weka.jar')
  default_config.set('tools', 'textcat', '%(bin)s')
  default_config.set('tools', 'libs', '%(bin)s')

  default_config.add_section('corpora')
  default_config.set('corpora', 'corpora', '~/data')
  default_config.set('corpora', 'reuters', '%(corpora)s/reut21578')
  default_config.set('corpora', 'eurogov', '%(corpora)s/eurogov')
  default_config.set('corpora', 'tcl', '%(corpora)s/TCL')
  default_config.set('corpora', 'udhr', '%(corpora)s/udhr/txt')
  default_config.set('corpora', 'wikipedia', '%(corpora)s/wikipedia')
  default_config.set('corpora', 'naacl2010-langid', '%(corpora)s/naacl2010-langid')

  default_config.add_section('logging')
  default_config.set('logging', 'console.level', 'info')
  default_config.set('logging', 'console.format', '%(name)s : %(message)s')
  default_config.set('logging', 'logfile', '')
  default_config.set('logging', 'logfile.level', 'debug')
  default_config.set('logging', 'logfile.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  default_config.add_section('random')
  default_config.set('random', 'seed', '83441363')
  default_config.set('random', 'allow_default_rng', 'False')

  return default_config

def write_configuration(config, file=DEFAULT_CONFIG_FILE):
  """ Write configuration to a file
  @param file File name or file-like object
  """
  if isinstance(file, str):
    file = open(file, 'w')
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

def update_configuration(config):
  """ Receives a config object, then scans hydrat for requirements,
  e.g. installed packages, tries to satisfy the requirements and 
  returns an updated configuration.
  """
  # TODO: 
  # Check for conflicting keys, and do something about it.
  # Might not actually be an error: More than one package might ask for java.
  import hydrat.classifier as c
  for klass in all_subclasses(c.abstract.Learner):
    requires = klass.requires
    for key in requires:
      toolpath = which(requires[key])
      logger.info("Resolved %s to %s", key, toolpath)
      if toolpath is not None:
        config.set('tools', key, toolpath)
  return config

logger = logging.getLogger("hydrat")
logger.setLevel(logging.DEBUG)

console_output = logging.StreamHandler()
console_output.setLevel(logging.CRITICAL)
logger.addHandler(console_output)

logfile_output = None

rng = None

LEVELS =\
  { 'debug': logging.DEBUG
  , 'info': logging.INFO
  , 'warning': logging.WARNING
  , 'error': logging.ERROR
  , 'critical': logging.CRITICAL
  }

def load_configuration(config):
  """ Process the hydrat configuration file. Should be invoked whenever the configuration changes.
      Right now this is mostly only useful for setting up logging.
  """
  global logger

  # Process options related to logging
  global console_output, logfile_output
  console_level = LEVELS.get(config.get('logging','console.level'), logging.NOTSET)
  console_formatter = logging.Formatter(config.get('logging','console.format',raw=True))
  console_output.setLevel(console_level)
  console_output.setFormatter(console_formatter)

  if config.get('logging','logfile'):
    logfile_output = logging.FileHandler(config.getpath('logging','logfile'), delay=True)
    logfile_level = LEVELS.get(config.get('logging','logfile.level'), logging.NOTSET)
    logfile_formatter = logging.Formatter(config.get('logging','logfile.format',raw=True))
    logfile_output.setLevel(logfile_level)
    logfile_output.setFormatter(logfile_formatter)
    logger.addHandler(logfile_output)
    

  local_logger = logging.getLogger('hydrat.load_configuration')
  # Process options related to random number management 
  global rng
  seed = config.getint('random','seed')
  rng = numpy.random.mtrand.RandomState(seed)
  local_logger.debug('Set random seed to %d', seed)
   


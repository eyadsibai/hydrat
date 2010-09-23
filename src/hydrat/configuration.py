import ConfigParser
import logging
import os
import sys
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

class DIR(object):
  def __init__(self, dirname):
    self.dirname = dirname

  def value(self):
    from hydrat import config
    search_paths = [ config.getpath('paths','corpora') , '~']
    return find_dir(self.dirname, search_paths)

class EXE(object):
  def __init__(self, filename):
    self.filename = filename
  
  def value(self):
    return which(self.filename)

class PACKAGE(object):
  """
  Intended for use with corpora.
  Needs to contain all the relevant information to locate if a package
  has been installed, or to obtain it if it has not been installed.
  Could be expanded to dealing with classifier packages as well.
  """
  pass

class Configurable(object):
  """
  Mixin for classes that are configurable.
  All subclasses of this class get scanned for key-value pairs
  that need to be added to the master configuration.
  """
  requires = {}

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
  """
  Recursively find all subclasses of a given class
  """
  subclasses = klass.__subclasses__()
  result = set(subclasses)
  for k in subclasses:
    result.update(all_subclasses(k))
  return result

def is_exe(fpath):
  return os.path.exists(fpath) and os.access(fpath, os.X_OK)

def which(program):
  # from http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
  logger.debug("which '%s'", program)
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


def find_dir(dirname, search_paths = ['~']):
  """
  Locate a directoy given its name.
  """
  logger.debug("find_dir '%s' in '%s'", dirname, str(search_paths))
  result = []
  class Found(Exception): pass
  def visit(arg, dn, names):
    if os.path.basename(dn) == dirname:
      arg.append(dn)
      raise Found
  try:
    for path in search_paths:
      os.path.walk(os.path.expanduser(path), visit, result)
  except Found:
    pass
  if len(result) != 0:
    return result[0]
  else:
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
  default_config.set('paths', 'corpora', '~/data')

  default_config.add_section('tools')
  default_config.set('tools', 'rainbow', '/usr/bin/rainbow')
  default_config.set('tools', 'weka', '/usr/bin/weka.jar')
  default_config.set('tools', 'textcat', '')
  default_config.set('tools', 'libs', '')
  default_config.set('tools', 'genia_data', '')
  default_config.set('tools', 'gibbslda', '')

  default_config.add_section('corpora')
  #default_config.set('corpora', 'udhr', '%(corpora)s/udhr/txt')
  # TODO: This stub is needed because NAACL2010 asks for the path at class instantiation.
  #       Should work around this via the Configurable API, so the object looks up its 
  #       local configuration before accessing the global one.
  default_config.set('corpora', 'naacl2010-langid', '')

  default_config.add_section('logging')
  default_config.set('logging', 'console.level', 'info')
  default_config.set('logging', 'console.format', '%(name)s : %(message)s')
  default_config.set('logging', 'logfile', '')
  default_config.set('logging', 'logfile.level', 'debug')
  default_config.set('logging', 'logfile.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  default_config.add_section('random')
  default_config.set('random', 'seed', '83441363')
  default_config.set('random', 'allow_default_rng', 'False')

  default_config.add_section('debug')
  default_config.set('debug', 'pdb_on_classifier_exception', 'False')
  default_config.set('debug', 'pdb_on_unhandled_exception', 'False')
  default_config.set('debug', 'clear_temp_files', 'True')

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

def update_configuration(config, rescan=False):
  """ Receives a config object, then scans hydrat for requirements,
  e.g. installed packages, tries to satisfy the requirements and 
  returns an updated configuration.
  """
  # TODO: 
  # Check for conflicting keys, and do something about it.
  # Might not actually be an error: More than one package might ask for java.
  import hydrat
  for klass in Configurable.__subclasses__():
    requires = klass.requires
    for section,key in requires:
      if config.has_option(section, key) and config.getpath(section,key) is not '':
        # Already have a setting
        existing = config.getpath(section, key)
        if rescan:
          toolpath = requires[(section,key)].value()
          if toolpath is not None and toolpath != existing:
            logger.info("%s --> %s (updated)", key, toolpath)
            config.set(section, key, toolpath)
          else:
            logger.info("%s --> %s (no update)", key, existing)
        else:
          logger.info("%s --> %s (existing configuration)", key, existing)
      else:
        toolpath = requires[(section,key)].value()
        if toolpath is not None:
          logger.info("%s --> %s (resolved)", key, toolpath)
          config.set(section, key, toolpath)
        else:
          logger.info("%s --> None (not found)", key)
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

  # Process options related to debugging
  # Based on http://ynniv.com/blog/2007/11/debugging-python.html
  def info(type, value, tb):
    if (#hasattr(sys, "ps1") or
        not sys.stderr.isatty() or 
        not sys.stdin.isatty()):
        # stdin or stderr is redirected, just do the normal thing
        original_hook(type, value, tb)
    else:
        # a terminal is attached and stderr is not redirected, debug 
        import traceback, pdb
        traceback.print_exception(type, value, tb)
        print
        pdb.pm()
        #traceback.print_stack()

  original_hook = sys.excepthook
  if config.getboolean('debug','pdb_on_unhandled_exception'):
    sys.excepthook = info
   


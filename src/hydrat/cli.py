"""
CLI interface for hydrat
Each function should consume the balance of the command-line arguments.
..todo: Make this nicer
"""
import configuration
import sys
import logging

logger = logging.getLogger(__name__)

def write_config(args):
  path = args[0]
  configuration.write_default_configuration(path)
  sys.exit(0)

from hydrat.dataset import check_dataset
def dataset_info(args):
  corpusname = args[0]
  try:
    import sys
    sys.path.append('.')
    exec('from %s import %s as ds' % tuple(corpusname.rsplit('.',1)))
    print ds()
  except ImportError, e:
    try:
      exec('from hydrat.corpora.%s import %s as ds' % tuple(corpusname.rsplit('.',1)))
      print ds()
    except ImportError, e:
      logger.debug(e)
      print "Unable to locate %s" % corpusname
    except TypeError, e:
      logger.debug(e)
      print "%s is not a dataset" % corpusname

  
# Commands supported by the hydrat CLI tool
commands=\
  { 'write_config': write_config
  , 'dataset_info': dataset_info
  }

def handle_commands(args):
  try:
    command = args[0]
  except IndexError:
    print "No command given!"
    
  if command in commands:
    commands[command](args[1:])
  else:
    parser.error('Unknown command %s'% command)


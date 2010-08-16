"""
CLI interface for hydrat
Each function should consume the balance of the command-line arguments.
..todo: Make this nicer
        Add commands:
          - Vizualize all the results in a store
          - Vizualize datasets in a store
          - Vizualize tasksets in a store
          - Export a model to ARFF
          - (?) Import a model from ARFF
           
"""
import os
import sys
import logging
import cmdln

import configuration

logger = logging.getLogger(__name__)

class HydratCmdln(cmdln.Cmdln):
  def do_configure(self, subcmd, opts, *args):
    """${cmd_name}: write a configuration file

    ${cmd_usage} 

    Writes the default configuration file to .hydratrc
    """
    if len(args) > 0:
      path = args[0]
    else:
      path = configuration.DEFAULT_CONFIG_FILE

    if not os.path.isabs(path):
      path = os.path.abspath(path)

    config = configuration.default_configuration()
    config = configuration.update_configuration(config)
    configuration.write_configuration(config, path)
    logger.info("Wrote configuration file to '%s'", path)

  @cmdln.alias("dsinfo")
  def do_dataset_info(self, subcmd, opts, dsname):
    """${cmd_name}: display basic information about a dataset 

    ${cmd_usage} 

    For example, given a subclass of dataset Foo in a module bar.py,
    we call ${cmd_name} as follows:

      ${name} ${cmd_name} bar.Foo
    """
    sys.path.append('.')
    try:
      exec('from %s import %s as ds' % tuple(dsname.rsplit('.',1)))
      print(ds())
    except ImportError, e:
      try:
        exec('from hydrat.corpora.%s import %s as ds' % tuple(dsname.rsplit('.',1)))
        print(ds())
      except ImportError, e:
        logger.debug(e)
        print("Unable to locate %s" % dsname)
      except TypeError, e:
        logger.debug(e)
        print("%s is not a dataset" % dsname)


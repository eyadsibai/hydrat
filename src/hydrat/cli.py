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
  @cmdln.option("-d", "--default", action="store_true", default=False,
                    help="write a default configuration (do not parse existing config file)")
  @cmdln.option("-r", "--rescan", action="store_true", default=False,
                    help="rescan paths")
  @cmdln.option("-i", "--include", action="append", default=[],
                    help="include additional configuration files")
  def do_configure(self, subcmd, opts, *args):
    """${cmd_name}: write a configuration file

    ${cmd_usage} 

    Writes the default configuration file to .hydratrc
    """
    if opts.default and len(opts.include) > 0:
      self.optparser.error('-d and -i are mutually exclusive')

    if len(args) > 0:
      path = args[0]
    else:
      path = configuration.DEFAULT_CONFIG_FILE

    if not os.path.isabs(path):
      path = os.path.abspath(path)

    if opts.default: 
      # overwrite with default
      config = configuration.default_configuration()
    else:
      # Read in the existing configuration, so we don't lose
      # user customizations.
      config = configuration.read_configuration(opts.include)
    config = configuration.update_configuration(config, rescan=opts.rescan)
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

  def do_browse(self, subcmd, opts, store_path):
    """${cmd_name}: browse an existing hdf5 store

    ${cmd_usage} 

    For example, given a store called 'store.h5', we call ${cmd_name} as follows:

      ${name} ${cmd_name} store.h5

    The browser is configured via a python module called 'browser_config.py'. 
    If present in the working directory, this module will supersede the default
    one supplied with hydrat, and allows the user to specify certain parameters
    of the browser's behaviour. See the default at hydrat.browser.browser_config
    for more details of what is configurable.
    """
    import cherrypy
    from hydrat.store import Store
    from hydrat.browser import StoreBrowser
    store = Store(store_path, 'r')
    import sys
    sys.path.append('.')
    try:
      import browser_config
    except ImportError:
      import hydrat.browser.browser_config as browser_config

    # Try to determine local IP address
    # from http://stackoverflow.com/questions/166506/finding-local-ip-addresses-in-python
    # TODO: Deal with possible failure, and/or make this configurable.
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("gmail.com",80))
    hostname = s.getsockname()[0]

    cherrypy.config.update({'server.socket_host': hostname})
    cherrypy.quickstart(StoreBrowser(store, browser_config))

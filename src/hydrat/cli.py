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
  @cmdln.option("-s", "--scan", action="append", default=[],
                    help="specify additional paths to scan")
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
    config = configuration.update_configuration(config, rescan=opts.rescan, scan=opts.scan)
    configuration.write_configuration(config, path)
    logger.info("Wrote configuration file to '%s'", path)

  @cmdln.option("-s", "--spaces",   action="store_true", default=False, help="merge spaces")
  @cmdln.option("-d", "--datasets", action="store_true", default=False, help="merge datasets")
  @cmdln.option("-t", "--tasksets", action="store_true", default=False, help="merge tasksets")
  @cmdln.option("-r", "--results",  action="store_true", default=False, help="merge results")
  def do_merge(self, subcmd, opts, src, dst):
    """${cmd_name}: merge two or more stores 

    ${cmd_usage} 
    
    To merge the contents of storeA.h5 into storeB.h5:

      ${name} ${cmd_name} storeA.h5 storeB.h5
    """
    from store import Store
    src_store = Store(src, 'r')
    dst_store = Store(dst, 'a')
    if not any((opts.spaces, opts.datasets, opts.tasksets, opts.results)):
      args = dict( do_spaces=True, do_datasets=True, do_tasksets=True, do_results=True )
    else:
      args = dict(\
               do_spaces   = opts.spaces, 
               do_datasets = opts.datasets,
               do_tasksets = opts.tasksets, 
               do_results  = opts.results,
               )
    dst_store.merge(src_store, **args)
    logger.info("Merged %s into %s", src, dst)


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

  @cmdln.option("-f", "--force",   action="store_true", default=False, help="force generation of all summaries")
  @cmdln.option("-d", "--delete",  action="store_true", default=False, help="delete summary nodes")
  def do_summary(self, subcmd, opts, store_path):
    """${cmd_name}: create summaries 

    ${cmd_usage} 
    """
    from store import Store
    import sys
    sys.path.append('.')
    try:
      import browser_config
    except ImportError:
      import hydrat.browser.browser_config as browser_config

    store = Store(store_path,'a')

    summary_fn = browser_config.summary_fn
    interpreter = browser_config.interpreter
    int_id = interpreter.__name__
    for tsr_id in store._resolve_TaskSetResults({}):
      if opts.delete:
        res = store.del_Summary(tsr_id, int_id)
        if res:
          print "Deleted summary", int_id, "from", tsr_id
        else:
          print "Failed to delete", int_id, "from", tsr_id
      else:
        if opts.force:
          missing_keys = set(summary_fn.keys)
        else:
          summary = store.get_Summary(tsr_id, int_id)
          missing_keys = set(summary_fn.keys) - set(summary)
        if len(missing_keys) > 0:
          result = store._get_TaskSetResult(tsr_id)
          summary_fn.init(result, interpreter)
          new_values = dict( (key, summary_fn[key]) for key in missing_keys )
          store.add_Summary(tsr_id, int_id, new_values, overwrite=opts.force) 
        print "Added", missing_keys, "to", tsr_id

  # TODO: Refactor against frameworks.offline and browser.results
  def do_output(self, subcmd, opts, store_path, output_path):
    """${cmd_name}: produce and upload summaries in html format

    ${cmd_usage} 
    """
    from store import Store
    from display.store import results2html
    from hydrat import config
    import sys
    import os.path
    import tempfile
    import updatedir
    import shutil
    sys.path.append('.')
    try:
      import browser_config
    except ImportError:
      # TODO: This has the side effect of failing if your browser_config fails with an importerror.
      import hydrat.browser.browser_config as browser_config

    store = Store(store_path)
    scratchP = tempfile.mkdtemp('output', dir=config.getpath('paths','scratch'))
    with open(os.path.join(scratchP, 'index.html'), 'w') as f:
      f.write(results2html(store, browser_config))

    updatedir.logger = logger
    print "Uploading..."
    updatedir.updatetree(scratchP, output_path, overwrite=True)
    if config.getboolean('debug','clear_temp_files'):
      shutil.rmtree(scratchP)
      

  @cmdln.option("-r", "--remote", action="store_true", default=False,
                    help="set up remote access to browser webapp")
  @cmdln.option("-b", "--nobrowse", action="store_true", default=False,
                    help="do not attempt to launch a webbrowser")
  @cmdln.option("-m", "--modify", action="store_true", default=False,
                    help="allow the store to be modified")
  @cmdln.option("-p", "--port", type='int', default=8080,
                    help="listen on port number")
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
    store = Store(store_path, 'a' if opts.modify else 'r')
    import sys
    sys.path.append('.')
    try:
      import browser_config
    except ImportError:
      import hydrat.browser.browser_config as browser_config

    # Try to determine local IP address
    # from http://stackoverflow.com/questions/166506/finding-local-ip-addresses-in-python
    import socket
    if opts.remote:
      s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      s.connect(("gmail.com",80))
      hostname = s.getsockname()[0]
    else:
      hostname = socket.gethostbyname(socket.gethostname())

    cherrypy.config.update({'server.socket_host': hostname, 'server.socket_port':opts.port})
    cherrypy.tree.mount(StoreBrowser(store, browser_config))
    cherrypy.engine.start()
    if not opts.nobrowse:
      import webbrowser
      webbrowser.open('http://%s:%d'%(hostname, opts.port))
    cherrypy.engine.block()

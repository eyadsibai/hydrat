"""
Twitter Language Identifier based on hydrat
by Marco Lui December 2010
"""

###
# Configuration Defaults
###

# Defaults for XMLRPC interface
HOST = "localhost"
PORT = 9008

# Configure where temporary files will be stored
TMP = '.langid' 

# URL to download pre-trained models from
DEFAULT_MODEL_URL = 'http://hum.csse.unimelb.edu.au/~mlui/langid/resources/hydrat/langidtool.tgz'


import sys

try:
  import hydrat
except ImportError:
  print "This tool requires hydrat to be installed. Please see http://code.google.com/p/hydrat/wiki/Installation"
  sys.exit(-1)

import os
import logging
import optparse

import hydrat.corpora.langid.naacl2010 as naacl2010
from hydrat.frameworks.online import OnlineFramework
from hydrat.result.interpreter import SingleHighestValue
from hydrat.classifier.meta.prototype import mean_prototypeL
from hydrat.classifier.knn import skew_1nnL

import hydrat.common.pb as pb

logger = logging.getLogger(__name__)

def fetch_model(url=DEFAULT_MODEL_URL):
  import tarfile
  tarpath = os.path.join(TMP, 'langidtool.tgz')
  if not os.path.exists(tarpath):
    import urllib
    try:
      urllib.urlretrieve(url, tarpath)
    except Exception:
      #TODO: catch more specific exceptions
      print "Unable to automatically download language models"
      print "Please download: %s" % DEFAULT_MODEL_URL
      print " and save it to: %s" % os.path.abspath(TMP)
      sys.exit(-1)
  f = tarfile.open(tarpath)
  f.extractall(TMP)

if __name__ == "__main__":
  parser = optparse.OptionParser()
  parser.add_option('-x','--xmlrpc',action='store_true', default=False, dest='xmlrpc')
  parser.add_option('--host', default=HOST, dest='host', help='hosti/ip to bind to')
  parser.add_option('--port', default=PORT, dest='port', help='port to listen on')
  parser.add_option('-v', action='count', dest='verbosity', help='increase verbosity (repeat for greater effect)')
  options, args = parser.parse_args()

  if options.verbosity:
    logging.basicConfig(level=max((5-options.verbosity)*10, 0))

  if not os.path.exists(TMP): 
    os.mkdir(TMP)
  modelpath = os.path.join(TMP, 'langidtool.h5')
  if not os.path.exists(modelpath):
    fetch_model()

  ds = naacl2010.Wikipedia()
  fw = OnlineFramework(ds, store=modelpath)
  
  fw.set_feature_spaces(['byte_bigram'])
  fw.set_class_space('iso639_1')
  fw.set_learner(mean_prototypeL(skew_1nnL()))
  import hydrat.classifier.nearest_prototype as np
  fw.set_interpreter(SingleHighestValue())

  # Silence progressbar
  pb.ENABLED = False
  if options.xmlrpc:
    print "Listening on %s:%d" % (options.host, options.port)
    fw.serve_xmlrpc(options.host, options.port)
  else:
    while True:
      try:
        print ">>>",
        text = raw_input()
      except Exception:
        break
      logger.debug("Classifying: '%s'", text)
      klass = fw.classify(text)
      print klass[0]

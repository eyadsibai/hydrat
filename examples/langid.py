#!/usr/bin/env python
"""
Twitter Language Identifier 
by Marco Lui December 2010

Copyright 2010 Marco Lui <saffsd@gmail.com>. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of the copyright holder.
############################################################################################
This software provides a command-line interface if invoked without 
options, and an xmlrpc interface if started with the '-x' option.

Please invoke the program with the -h option for additional help messages.

Upon first running, the program will attempt to download language models from a
pre-configured URL. If it is unable to do so, it will display the URL and ask the
user to manually obtain the required file.

Here is an example use of the xmlrpc interface via a python program:

import xmlrpclib
import json

proxy = xmlrpclib.ServerProxy('http://localhost:9008')
config = json.loads(proxy.configuration())
print config
print proxy.classify('We can classify one string at a time')
print proxy.classify('Si puo anche classificare testo in italiano')
print proxy.classify([
  'Alternatively, a list of strings can be provided',
  'and a list of classifications will be returned as a result',
])
############################################################################################
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
  print "This tool requires hydrat to be installed.",
  print "Please see http://code.google.com/p/hydrat/wiki/Installation"
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
  parser.add_option('--host', default=HOST, dest='host', help='host/ip to bind to')
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

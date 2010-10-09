#!/usr/bin/env python
"""
Wrapper for google's langid API
based on code from http://stackoverflow.com/questions/1136604/how-do-i-use-the-json-google-translate-api
"""
import urllib
import simplejson
import time
import random
import logging
logger = logging.getLogger(__name__)

class GoogleAPI(object):
  base_url='http://ajax.googleapis.com/ajax/services/language/detect?'

  def __init__( self, sleep=5, simulate=False, retry = 60 ):
    self.sleep = sleep # Time between requests
    self.simulate = simulate
    self.retry = retry

  def classify(self, text):
    if isinstance(text, unicode): text = text.encode('utf-8')
    if self.simulate:
      json = {'responseData':{'language':'en'}}
    else:
      data = urllib.urlencode({'v':1.0,'ie': 'UTF8', 'q': text})
      search_results = urllib.urlopen(self.base_url+data)
      json = simplejson.loads(search_results.read())
    while json['responseData'] is None:
      logger.warning(json)
      logger.warning("Got a None response, retrying in %d seconds", retry)
      time.sleep(self.retry)
      self.retry *= 2
      try:
        json = simplejson.loads(search_results.read())
      except ValueError:
        json = {'responseData': None}
    result = json['responseData']['language']
    return result

  def batch_classify(self, texts):
    result = []
    for text in texts:
      result.append(self.classify(text))
      time.sleep(random.random() * self.sleep)
    return result
      


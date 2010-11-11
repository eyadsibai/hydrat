#!/usr/bin/env python
"""
Wrapper for google's langid API
based on code from http://stackoverflow.com/questions/1136604/how-do-i-use-the-json-google-translate-api
"""
#TODO: Move the body of this functionality into hydrat.wrapper?
import urllib
import json
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
      response = {'responseData':{'language':'en'}}
    else:
      data = urllib.urlencode({'v':1.0,'ie': 'UTF8', 'q': text})
      search_results = urllib.urlopen(self.base_url+data)
      response = json.loads(search_results.read())
    while response['responseData'] is None:
      logger.warning(response)
      logger.warning("Got a None response, retrying in %d seconds", self.retry)
      time.sleep(self.retry)
      self.retry *= 2
      search_results = urllib.urlopen(self.base_url+data)
      try:
        response = json.loads(search_results.read())
      except ValueError:
        response = {'responseData': None}
    result = response['responseData']['language']
    return result

  def batch_classify(self, texts):
    result = []
    for text in texts:
      result.append(self.classify(text))
      time.sleep(random.random() * self.sleep)
    return result
      


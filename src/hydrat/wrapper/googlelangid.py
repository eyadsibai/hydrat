#!/usr/bin/env python
"""
Wrapper for google's langid API
based on code from http://stackoverflow.com/questions/1136604/how-do-i-use-the-json-google-translate-api
"""
import urllib
import json
import time
import random
import logging
logger = logging.getLogger(__name__)

class GoogleLangid(object):
  base_url='http://ajax.googleapis.com/ajax/services/language/detect?'

  def __init__( self, sleep=5, simulate=False, retry = 60, apikey=None ):
    self.sleep = sleep # Time between requests
    self.simulate = simulate
    self.retry = retry
    self.apikey = apikey

  def classify(self, text):
    if isinstance(text, unicode): text = text.encode('utf-8')
    if self.simulate:
      response = {'responseData':{'language':'en'}}
    else:
      query = {'v':1.0,'ie': 'UTF8', 'q': text}
      if self.apikey is not None:
        query['key'] = self.apikey
      data = urllib.urlencode(query)
      search_results = urllib.urlopen(self.base_url+data)
      response = json.loads(search_results.read())
    retry = self.retry
    while response['responseData'] is None:
      logger.warning(response)
      logger.warning("Got a None response, retrying in %d seconds", retry)
      time.sleep(retry)
      retry *= 2
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
      


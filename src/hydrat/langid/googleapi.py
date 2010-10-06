#!/usr/bin/env python
"""
Wrapper for google's langid API
based on code from http://stackoverflow.com/questions/1136604/how-do-i-use-the-json-google-translate-api
"""
import urllib
import simplejson
import time

class GoogleAPI(object):
  base_url='http://ajax.googleapis.com/ajax/services/language/detect?'

  def __init__( self, sleep = 0.1 ):
    self.sleep = sleep # Time between requests

  def classify(self, text):
    if isinstance(text, unicode): text = text.encode('utf-8')
    data = urllib.urlencode({'v':1.0,'ie': 'UTF8', 'q': text})
    search_results = urllib.urlopen(self.base_url+data)
    json = simplejson.loads(search_results.read())
    result = json['responseData']['language']
    print result, text,
    return result

  def batch_classify(self, texts):
    result = []
    for text in texts:
      result.append(self.classify(text))
      time.sleep(self.sleep)
    return result
      


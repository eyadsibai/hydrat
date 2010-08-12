from __future__ import with_statement
from sgmllib import SGMLParser
from hydrat.dataset.encoded import BagOfWords
from hydrat import config
import logging
from time import time
from collections import defaultdict
import os

logger = logging.getLogger('hydrat.datasets.Reuters21578')

class ReutersParser(SGMLParser):
  def __init__(self, verbose=0):
    SGMLParser.__init__(self,verbose)
    self.reset()

  def reset(self):
    SGMLParser.reset(self)
    self.docmap = {}
    self.classmap = {}
    self.inReutersBlock = False
    self.inTopic = False
    self.addToTopics = False
    self.keepContent = False
    self.content = ""
    self.lewissplit = {"TRAIN":[], "TEST":[], "NOT-USED":[]}

  def start_reuters(self,attrs):
    attrd = dict(attrs)
    self.inReutersBlock = True
    self.currentid = attrd["newid"]
    self.classmap[self.currentid] = []
    self.lewissplit[attrd["lewissplit"]].append(self.currentid)

  def start_topics(self, attrs):
    if self.inReutersBlock:
      self.inTopic = True

  def end_topics(self):
    self.inTopic = False

  def start_d(self, attrs):
    if self.inTopic:
      self.addToTopics = True

  def end_d(self):
    if self.inTopic:
      self.addToTopics = False

  def start_text(self,attrs):
    if self.inReutersBlock:
      self.keepContent = True

  def end_text(self):
    self.keepContent = False
 
  def handle_data(self, data):
    if self.keepContent:
      self.content += data
    if self.addToTopics:
      self.classmap[self.currentid].append(data)

  def end_reuters(self):
    if self.inReutersBlock:
      self.docmap[self.currentid] = self.content
    self.content = ""
    self.currentid = None
    self.inReutersBlock = False
    self.keepContent = False

  def run(self):
    self.reset()
    logger.debug("Parsing reuters data set")
    start_time = time()
    for i in range(0,22):
      with open(os.path.join(config.getpath('corpora', 'reuters'),"reut2-"+"%03d"%i+".sgm")) as input:
        self.feed(input.read())
    time_taken = time() - start_time
    logger.debug("Completed parsing in %3.2f seconds" % time_taken)
    return (self.docmap, self.classmap, self.lewissplit)


class Reuters21578(BagOfWords):
  __name__ = 'Reuters21578'
  __parser = None

  @property
  def _parser(self):
    if self.__parser is None:
      self.__parser = ReutersParser()
      self.__parser.run()
    return self.__parser

  def encodings(self):
    return defaultdict(lambda:'ascii')

  def text(self):
    p = self._parser
    return p.docmap
    
  def cm_reuters21578_topics(self):
    p = self._parser
    return p.classmap
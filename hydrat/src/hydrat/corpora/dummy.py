from hydrat.dataset.text import ByteUBT
from hydrat.dataset.encoded import CodepointUBT
from collections import defaultdict

class dummy(ByteUBT, CodepointUBT):
  """Dummy backend for development use"""
  __name__ = "dummy"
  words = [u"test", u"exam", u"eggs", u"spam", u"blah"]

  def __init__(self, max_times = 100):
    ByteUBT.__init__(self)
    CodepointUBT.__init__(self)
    self.max_times = max_times
    self.__name__ += str(max_times)

  def encodings(self):
    return defaultdict(lambda:'utf-8')

  def text(self):
    docmap = {}
    for i in xrange(len(self.words)):
      for j in xrange(self.max_times):
        docmap["%04d"%(i * self.max_times + j)] = (self.words[i].encode('utf8') + " ") * (j+1)
    return docmap 

  def cm_dummy_default(self):
    classmap = {}
    for i in xrange(len(self.words)):
      for j in xrange(self.max_times):
        classmap["%04d"%(i * self.max_times + j)] = [u'class' + unicode(i) ]
    return classmap

class unicode_dummy(dummy):
  """Unicode dummy dataset"""
  __name__ = "dummy-unicode"
  words = [u"\N{POUND SIGN}pound",u'\N{BLACK STAR}blackstar',u'\N{WHITE STAR}whitestar',u'\N{LIGHTNING}lightning',u'\N{COMET}comet']

class single_char_dummy(dummy):
  __name__ = "dummy-single"
  words = [u'A',u'B',u'C',u'D',u'E']

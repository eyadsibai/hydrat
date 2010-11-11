from collections import defaultdict
from hydrat.dataset.text import TextDataset
from hydrat.common.pb import ProgressIter
import hydrat.common.extractors as ext

class EncodedTextDataset(TextDataset):
  def __init__(self):
    TextDataset.__init__(self)
    self.__encodings = None

  def encodings(self):
     """
     Return a dictionary from instance identifiers
     to a string representing the encoding of the instance
     """
     raise NotImplementedError

  def _encodings(self):
    if self.__encodings is None:
      self.__encodings = self.encodings()
    return self.__encodings

  def ts_codepoint(self):
    text = self.tokenstream('byte')
    encodings = self._encodings()
    u = {}
    for instance_id in text:
      e = encodings[instance_id]
      try:
        u[instance_id] = text[instance_id].decode(e)
      except UnicodeDecodeError:
        self.logger.warning("Error decoding '%s' with codec '%s'", instance_id, e)
        self.logger.warning("Replacing undecodable characters")
        u[instance_id] = unicode(text[instance_id], encoding=e, errors='replace')
    return u

class UTF8(EncodedTextDataset):
  """mixin for a dataset that is entirely UTF8-encoded"""
  def encodings(self):
    return defaultdict(lambda:'utf-8')

class ASCII(EncodedTextDataset):
  """mixin for a dataset that is entirely ascii-encoded"""
  def encodings(self):
    return defaultdict(lambda:'ascii')

try:
  import chardet
  class AutoEncoding(EncodedTextDataset):
    """mixin for using chardet to autodetect character encodings""" 
    def encodings(self):
      text = self._text()
      e = dict()
      for i in self.instance_ids:
        enc = chardet.detect(text[i])
        self.logger.debug("Detected encoding '%s'(conf:%.2f) for '%s'",enc['encoding'],enc['confidence'],i)
        if enc['encoding'] == None:
          # We get a None back for empty strings, so just handle it by saying ascii
          e[i]= 'ascii'
        else:
          e[i] = enc['encoding']
      return e
except ImportError:
  pass


class CodepointUnigram(EncodedTextDataset):
  def fm_codepoint_unigram(self): return self.features('codepoint', ext.unigram)

class CodepointBigram(EncodedTextDataset):
  def fm_codepoint_bigram(self): return self.features('codepoint', ext.bigram)

class CodepointTrigram(EncodedTextDataset):
  def fm_codepoint_trigram(self): return self.features('codepoint', ext.trigram)

class CodepointUBT(CodepointUnigram, CodepointBigram, CodepointTrigram): pass

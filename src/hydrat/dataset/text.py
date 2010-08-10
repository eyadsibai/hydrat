from hydrat.dataset import Dataset
from hydrat.preprocessor.model.inducer import invert_text
from hydrat.common.tokenizers import NGram
from hydrat.common.pb import ProgressIter

class TextDataset(Dataset):
  """ Base class for datasets where instances can be represented
      as single string. Ideal for traditional text classification
      tasks.

      The only requirement for subclassing TextDataset is that the 
      subclass must implement the text method, which returns a 
      dictionary mapping from the instance identifier to the
      text of the instance.
  """
  def __init__(self):
    Dataset.__init__(self)
    self.__text = None

  def _text(self):
    if self.__text is None:
      self.__text = self.text()
    return self.__text

  def text(self):
    """
    Return a dictionary from instance identifiers
    to the content of the instance in a string 
    This should be a normal byte string.
    """
    raise NotImplementedError

  def text_token(self, tokenizer, text=None):
    """
    Generate feature map by applying a tokenizer to the raw
    text and then producing token counts
    """
    if text is None: text = self._text()
    fm = {}

    for instance_id in ProgressIter(text, label="Processing Documents"):
      fm[instance_id] = invert_text(text[instance_id], tokenizer)
      if len(fm[instance_id]) == 0:
        self.logger.warning( "Tokenizer did not return any tokens for %s", instance_id )


    return fm

class ByteUnigram(TextDataset):
  def fm_byte_unigram(self): return self.text_token(NGram(1))

class ByteBigram(TextDataset):
  def fm_byte_bigram(self): return self.text_token(NGram(2))

class ByteTrigram(TextDataset):
  def fm_byte_trigram(self): return self.text_token(NGram(3))

class ByteUBT(ByteUnigram, ByteBigram, ByteTrigram): pass

class ByteQuadgram(TextDataset):
  def fm_byte_quadgram(self): return self.text_token(NGram(4))

class BytePentagram(TextDataset):
  def fm_byte_pentagram(self): return self.text_token(NGram(5))


from hydrat.dataset.text import TextDataset
from hydrat.common.tokenizers import bag_of_words
import hydrat.common.extractors as ext
from hydrat.common.pb import ProgressIter


class WhitespaceWords(TextDataset):
  def ts_word(self):
    text = self.tokenstream('byte')
    streams = dict( (i,bag_of_words(text[i])) for i in ProgressIter(text,'Whitespace Words') )
    return streams

class BagOfWords(WhitespaceWords):
  def fm_word_unigram(self): return self.features('word', ext.unigram)

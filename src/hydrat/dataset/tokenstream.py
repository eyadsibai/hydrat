from hydrat.dataset.encoded import EncodedTextDataset
from hydrat.preprocessor.tokenstream.porterstem import PorterStemTagger
from hydrat.common.pb import ProgressIter

class PorterStem(EncodedTextDataset):
  def ts_porterstemmer(self):
    text = self._unicode()
    stemmer = PorterStemTagger()
    streams = dict( (i,stemmer.process(text[i])) for i in ProgressIter(text,'Porter Stemming') )
    return streams

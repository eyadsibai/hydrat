from hydrat.dataset.encoded import EncodedTextDataset
from hydrat.preprocessor.tokenstream.porterstem import PorterStemTagger
from hydrat.common.pb import ProgressIter

#TODO: Keep partial tokenstreams in shelve objects in scratch space - this will avoid recomputation if it cuts
# out halfway somehow.
class PorterStem(EncodedTextDataset):
  def ts_porterstemmer(self):
    text = self._unicode()
    stemmer = PorterStemTagger()
    streams = dict( (i,stemmer.process(text[i].encode('utf8'))) for i in ProgressIter(text,'Porter Stemmer') )
    return streams

from hydrat.preprocessor.tokenstream.genia import GeniaTagger
class Genia(EncodedTextDataset):
  def ts_genia(self):
    text = self._unicode()
    stemmer = GeniaTagger()
    streams = dict( (i,stemmer.process(text[i].encode('utf8'))) for i in ProgressIter(text,'GENIA Tagger') )
    return streams


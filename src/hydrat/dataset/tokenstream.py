from hydrat.dataset.text import TextDataset
from hydrat.preprocessor.tokenstream.porterstem import PorterStemTagger
from hydrat.common.pb import ProgressIter
from hydrat.wrapper.genia import GeniaTagger

#TODO: Keep partial tokenstreams in shelve objects in scratch space 
#      - this will avoid recomputation if it cuts out halfway somehow.
 
class PorterStem(TextDataset):
  def ts_porterstemmer(self):
    text = self.tokenstream('byte')
    stemmer = PorterStemTagger()
    streams = dict( (i,stemmer.process(text[i])) for i in ProgressIter(text,'Porter Stemmer') )
    return streams

class Genia(TextDataset):
  def ts_genia(self):
    text = self.tokenstream('byte')
    stemmer = GeniaTagger()
    streams = dict( (i,stemmer.process(text[i])) for i in ProgressIter(text,'GENIA Tagger') )
    return streams


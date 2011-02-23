from hydrat.dataset.encoded import EncodedTextDataset
from hydrat.preprocessor.tokenstream.porterstem import PorterStemTagger
from hydrat.common.pb import ProgressIter
from hydrat.wrapper.genia import GeniaTagger
from hydrat import config
from hydrat.configuration import Configurable, EXE, DIR

#TODO: Keep partial tokenstreams in shelve objects in scratch space 
#      - this will avoid recomputation if it cuts out halfway somehow.
 
class PorterStem(EncodedTextDataset):
  def ts_porterstemmer(self):
    text = self.ts_codepoint()
    stemmer = PorterStemTagger()
    streams = dict( (i,stemmer.process(text[i].encode('utf8'))) for i in ProgressIter(text,'Porter Stemmer') )
    return streams

class Genia(Configurable, EncodedTextDataset):
  requires =\
    { ('tools','genia')      : EXE('geniatagger')
    , ('tools','genia_data') : DIR('geniatagger-3.0.1')
    }

  def ts_genia(self):
    text = self.ts_codepoint()
    tagger_exe = config.getpath('tools','genia')
    genia_path = config.getpath('tools','genia_data')
    stemmer = GeniaTagger(tagger_exe, genia_path)
    streams = dict( (i,stemmer.process(text[i].encode('utf8'))) for i in ProgressIter(text,'GENIA Tagger') )
    return streams


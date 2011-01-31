import time
import logging
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.result.result import Result
from hydrat.store import Store
from hydrat.preprocessor.model.inducer import map2matrix 
from hydrat.wrapper.textcat import TextCat

from hydrat import config
from hydrat.configuration import Configurable, EXE, DIR
from hydrat.task.sampler import membership_vector

logger = logging.getLogger(__name__)

class TextCatConfig(Configurable):
  requires =\
    { 
    ('tools', 'textcat')         : EXE('text_cat'),
    ('tools', 'textcat-models')  : DIR('LM'),
    }

def identity(x): return x

textcat2iso639_1_assoc = [
  ('afrikaans', 'af'),
  ('albanian', 'sq'),
  ('amharic-utf', 'am'),
  ('arabic-iso8859_6', 'ar'),
  ('arabic-windows1256', 'ar'),
  ('armenian', 'hy'),
  ('basque', 'eu'),
  ('belarus-windows1251', 'be'),
  ('bosnian', 'bs'),
  ('breton', 'br'),
  ('bulgarian-iso8859_5', 'bg'),
  ('catalan', 'ca'),
  ('chinese-big5', 'zh'),
  ('chinese-gb2312', 'zh'),
  ('croatian-ascii', 'hr'),
  ('czech-iso8859_2', 'cs'),
  ('danish', 'da'),
  ('dutch', 'nl'),
  ('english', 'en'),
  ('esperanto', 'eo'),
  ('estonian', 'et'),
  ('finnish', 'fi'),
  ('french', 'fr'),
  ('frisian', 'fy'),
  ('georgian', 'ka'),
  ('german', 'de'),
  ('greek-iso8859-7', 'el'),
  ('hebrew-iso8859_8', 'he'),
  ('hindi', 'hi'),
  ('hungarian', 'hu'),
  ('icelandic', 'is'),
  ('indonesian', 'id'),
  ('irish', 'ga'),
  ('italian', 'it'),
  ('japanese-euc_jp', 'ja'),
  ('japanese-shift_jis', 'ja'),
  ('korean', 'ko'),
  ('latin', 'la'),
  ('latvian', 'lv'),
  ('lithuanian', 'lt'),
  ('malay', 'ms'),
  ('manx', 'gv'),
  ('marathi', 'mr'),
  ('mingo', 'UNKNOWN'),
  ('nepali', 'ne'),
  ('norwegian', 'no'),
  ('persian', 'fa'),
  ('polish', 'pl'),
  ('portuguese', 'pt'),
  ('quechua', 'qu'),
  ('romanian', 'ro'),
  ('rumantsch', 'rm'),
  ('russian-iso8859_5', 'ru'),
  ('russian-koi8_r', 'ru'),
  ('russian-windows1251', 'ru'),
  ('sanskrit', 'sa'),
  ('scots_gaelic', 'gd'),
  ('scots', 'UNKNOWN'), #sco
  ('serbian-ascii', 'sr'),
  ('slovak-ascii', 'sk'),
  ('slovak-windows1250', 'sk'),
  ('slovenian-ascii', 'sk'),
  ('slovenian-iso8859_2', 'sk'),
  ('spanish', 'es'),
  ('swahili', 'sw'),
  ('swedish', 'sv'),
  ('tagalog', 'tl'),
  ('tamil', 'ta'),
  ('thai', 'th'),
  ('turkish', 'tr'),
  ('ukrainian-koi8_u', 'uk'),
  ('vietnamese', 'vi'),
  ('welsh', 'cy'),
  ('yiddish-utf', 'yi'),
]

textcat2iso639_1_dict = dict(textcat2iso639_1_assoc)

def textcat2iso639_1(klass):
  return textcat2iso639_1_dict.get(klass, 'UNKNOWN')

def train_textcat(ds, tokenstream, class_space):
  logger.info('train textcat')
  cat = TextCat(
    config.getpath('tools','textcat'), 
    scratch=config.getpath('paths','scratch'),
    modelpath=None,
  )
  ts = ds.tokenstream(tokenstream)
  cm = ds.classmap(class_space)
  pairs = [ (ts[i], cm[i][0]) for i in ds.instance_ids ]

  start = time.time()
  cat.train(pairs)
  train_time = time.time() - start

  cat.metadata = dict(
    dataset = ds.__name__, 
    instance_space = ds.instance_space, 
    tokenstream=tokenstream, 
    class_space=class_space,
    train_time=train_time,
    )
  cat.spacemap = identity
  return cat

def default_textcat():
  cat = TextCat(
    config.getpath('tools','textcat'), 
    scratch=config.getpath('paths','scratch'),
    modelpath=config.getpath('tools','textcat-models'),
  )

  cat.metadata = dict(
    dataset = 'textcat',
    instance_space = 'textcat', 
    tokenstream='byte', 
    class_space='iso639_1',
    train_time=None,
    )
  cat.spacemap = textcat2iso639_1
  return cat


def do_textcat(cat, ds, classlabels, instance_labels):
  # TODO Clean up the handling of the spacemap!
  spacemap = cat.spacemap
  logger.info('do textcat')
  class_space = cat.metadata['class_space']
  tokenstream = cat.metadata['tokenstream']
  md = dict(\
    class_space  = class_space,
    dataset      = cat.metadata['dataset'],
    eval_dataset = ds.__name__,
    instance_space = cat.metadata['instance_space'],
    eval_space   = ds.instance_space,
    learner      = 'textcat',
    learner_params = dict(tokenstream=tokenstream, spacemap=spacemap.__name__),
    )

  ts = ds.tokenstream(tokenstream)
  start = time.time()
  ids, texts = zip(*ts.items())
  klass = cat.batch_classify(texts)
  class_map = {}
  for id, cl in zip(ids, klass):
    class_map[id] = [ spacemap(cl) ]
  test_time = time.time() - start
  

  cl = map2matrix( class_map, ds.instance_ids, classlabels )
  gs = map2matrix( ds.classmap(class_space), ds.instance_ids, classlabels )

  result_md = dict(md)
  result_md['learn_time'] = cat.metadata['train_time']
  result_md['classify_time'] = test_time
  instance_indices = membership_vector(instance_labels, ds.instance_ids)
  result = Result(gs, cl, instance_indices, result_md )
  tsr = TaskSetResult( [result], md )
  return tsr

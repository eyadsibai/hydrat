import time
import logging
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.result.result import Result
from hydrat.store import Store
from hydrat.preprocessor.model.inducer import map2matrix 
from hydrat.wrapper.textcat import TextCat

from hydrat import config
from hydrat.configuration import Configurable, EXE
from hydrat.task.sampler import membership_vector

logger = logging.getLogger(__name__)

class TextCatConfig(Configurable):
  requires =\
    { ('tools', 'textcat')    : EXE('text_cat')
    }

def train_textcat(ds, tokenstream, class_space):
  logger.info('train textcat')
  cat = TextCat(config.getpath('tools','textcat'), config.getpath('paths','scratch'))
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
  return cat

def do_textcat(cat, ds, classlabels, instance_labels):
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
    learner_params = dict(tokenstream=tokenstream),
    )

  ts = ds.tokenstream(tokenstream)
  start = time.time()
  ids, texts = zip(*ts.items())
  klass = cat.batch_classify(texts)
  class_map = {}
  for id, cl in zip(ids, klass):
    class_map[id] = [ cl ]
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

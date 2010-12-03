import time
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.result.result import Result
from hydrat.store import Store
from hydrat.preprocessor.model.inducer import map2matrix 
from hydrat.wrapper.textcat import TextCat

from hydrat import config
from hydrat.configuration import Configurable, EXE
from hydrat.task.sampler import membership_vector

class TextCatConfig(Configurable):
  requires =\
    { ('tools', 'textcat')    : EXE('text_cat')
    }

def do_textcat(train_ds, test_ds, tokenstream, class_space, classlabels, instance_labels):
  md = dict(\
    class_space  = class_space,
    dataset      = train_ds.__name__,
    eval_dataset = test_ds.__name__,
    instance_space = train_ds.instance_space,
    eval_space   = test_ds.instance_space,
    learner      = 'textcat',
    learner_params = dict(tokenstream=tokenstream),
    )

  cat = TextCat(config.getpath('tools','textcat'), config.getpath('paths','scratch'))
  train_ts = train_ds.tokenstream(tokenstream)
  train_cm = train_ds.classmap(class_space)
  pairs = [ (train_ts[i], train_cm[i][0]) for i in train_ds.instance_ids ]

  start = time.time()
  cat.train(pairs)
  train_time = time.time() - start

  test_ts = test_ds.tokenstream(tokenstream)
  class_map = {}
  start = time.time()
  for id in test_ts:
    klass = cat.classify(test_ts[id])
    class_map[id] = [ klass ]
  test_time = time.time() - start
  

  cl = map2matrix( class_map, test_ds.instance_ids, classlabels )
  gs = map2matrix( test_ds.classmap(class_space), test_ds.instance_ids, classlabels )

  result_md = dict(md)
  result_md['learn_time'] = train_time
  result_md['classify_time'] = test_time
  instance_indices = membership_vector(instance_labels, test_ds.instance_ids)
  result = Result(gs, cl, instance_indices, result_md )
  tsr = TaskSetResult( [result], md )
  return tsr

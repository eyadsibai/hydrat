import time
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.result.result import Result
from hydrat.preprocessor.model.inducer import map2matrix 
from hydrat.wrapper.langdetect import LangDetect
from hydrat.common.pb import ProgressBar, get_widget

from hydrat import config
from hydrat.configuration import Configurable, EXE, DIR, FILE
from hydrat.task.sampler import membership_vector

class LangDetectConfig(Configurable):
  requires={
    ('tools','java-bin')             : EXE('java'),
    ('tools','langdetect')           : FILE('langdetect.jar'),
    ('tools','langdetect-profiles')  : DIR('profiles'),
    }

def do_langdetect(store, test_ds, tokenstream, class_space, spacemap, batchsize=100, store_result = True):
  instance_labels = store.get_Space(test_ds.instance_space)
  classlabels = store.get_Space(class_space)
  md = dict(\
    class_space  = class_space,
    dataset      = 'LangDetect',
    eval_dataset = test_ds.__name__,
    instance_space = 'LangDetect',
    eval_space   = test_ds.instance_space,
    learner      = 'LangDetect',
    learner_params = dict(tokenstream=tokenstream, spacemap=spacemap.__name__, batchsize=batchsize),
    )

  if store.has_TaskSetResult(md):
    return store.get_TaskSetResult(md)
  else:
    cat = LangDetect(
      config.getpath('tools','java-bin'), 
      config.getpath('tools','langdetect'), 
      config.getpath('tools','langdetect-profiles'), 
      config.getpath('paths','scratch'),
      batchsize = batchsize,
      )

    test_ts = test_ds.tokenstream(tokenstream)
    class_map = {}
    start = time.time()

    with ProgressBar(widgets=get_widget('LangDetect'),maxval=len(test_ts)) as pb:
      klass = cat.batch_classify(test_ts.values(), callback=pb.update)

    for id, cl in zip(test_ts, klass):
      class_map[id] = [ spacemap(cl) ]
    test_time = time.time() - start
    
    cl = map2matrix( class_map, test_ds.instance_ids, classlabels )
    gs = map2matrix( test_ds.classmap(class_space), test_ds.instance_ids, classlabels )

    result_md = dict(md)
    result_md['learn_time'] = None
    result_md['classify_time'] = test_time
    instance_indices = membership_vector(instance_labels, test_ds.instance_ids)
    result = Result(gs, cl, instance_indices, result_md )
    tsr = TaskSetResult( [result], md )
    if store_result:
      store.new_TaskSetResult(tsr)
    return tsr

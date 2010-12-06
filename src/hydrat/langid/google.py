import random
import csv
import time
import datetime
import os.path
import numpy
import hydrat
from hydrat.common.pb import ProgressIter
from hydrat.langid import goog2iso639_1
from hydrat.wrapper.googlelangid import GoogleLangid
from hydrat.preprocessor.model.inducer import map2matrix 
from hydrat.store import Store
from hydrat.result.result import Result
from hydrat.result.tasksetresult import TaskSetResult

def goog_langid(ds, tokenstream, key=None):
  cat = GoogleLangid(retry=300, apikey=key)
  ts = ds.tokenstream(tokenstream)
  filename = 'google-%s-%s' % (tokenstream, ds.__name__)
  path = os.path.join(hydrat.config.getpath('paths','scratch'), filename)
  obtained = {}
  if os.path.exists(path):
    with open(path) as f:
      reader = csv.reader(f, delimiter='\t')
      for row in reader:
        obtained[row[0]] = row[1]

  with open(path, 'a') as f:
    writer = csv.writer(f, delimiter='\t')
    for key in ProgressIter(ts, label='Google-Langid'):
      if key in obtained: continue
      text = ts[key]
      sleep_len = random.uniform(0,10)
      now = datetime.datetime.now().isoformat()
      pred_lang = cat.classify(text)
      print sleep_len, pred_lang, now, text,
      time.sleep(sleep_len) # sleep for up to 3 seconds
      writer.writerow((key, pred_lang, now, text.strip()))
      f.flush()
  return obtained

def do_google(test_ds, tokenstream, class_space, classlabels, spacemap, key=None):
  md = dict(\
    class_space  = class_space,
    dataset      = 'GoogleLangid',
    eval_dataset = test_ds.__name__,
    instance_space = 'GoogleLangid',
    eval_space   = test_ds.instance_space,
    learner      = 'GoogleLangid',
    learner_params = dict(tokenstream=tokenstream, spacemap=spacemap.__name__)
    )

  preds = goog_langid(test_ds, tokenstream, key=key)
  for key in preds:
    preds[key] = [spacemap(preds[key])]

  cl = map2matrix( preds, test_ds.instance_ids, classlabels )
  gs = map2matrix( test_ds.classmap(class_space), test_ds.instance_ids, classlabels )

  result_md = dict(md)
  result_md['learn_time'] = None
  result_md['classify_time'] = None

  # We always use all instances
  instance_indices = numpy.ones(len(test_ds.instance_ids), dtype='bool')
  result = Result(gs, cl, instance_indices, result_md )
  tsr = TaskSetResult( [result], md )
  return tsr

      

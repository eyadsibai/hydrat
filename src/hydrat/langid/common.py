import numpy
import time

from hydrat.datamodel import TaskSetResult, Result
from hydrat.common.pb import ProgressBar, get_widget
from hydrat.common.mapmatrix import map2matrix

class External(TaskSetResult):
  def __init__(self, classifier, proxy):
    self.classifier = classifier
    self.proxy = proxy

  @property
  def metadata(self):
    keys = ['class_space','dataset','instance_space','learner','learner_params']
    md = dict()
    for key in keys:
      md[key] = self.classifier.metadata.get(key, None)
    md['eval_dataset'] = self.proxy.dsname
    md['eval_space']   = self.proxy.instance_space
    return md

  @property
  def results(self):
    proxy = self.proxy
    with ProgressBar(widgets=get_widget(self.classifier.__class__.__name__),maxval=len(self.proxy.instancelabels)) as pb:
      start = time.time()
      klass = self.classifier.classify_batch(self.proxy.tokenstream, callback=pb.update)
      test_time = time.time() - start

    class_map = dict(zip(proxy.instancelabels, klass))

    cl = map2matrix(class_map, proxy.instancelabels, proxy.classlabels)
    gs = proxy.classmap.raw
    instance_indices = numpy.arange(len(proxy.instancelabels))
    md = dict(learn_time=None, classify_time=test_time)
    return [ Result(gs, cl, instance_indices, md ) ]

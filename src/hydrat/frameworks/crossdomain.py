import numpy

from hydrat.frameworks.common import Framework
from hydrat.result.result import Result
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.common.decorators import replace_with_result

# TODO: Allow for feature weighting and selection
 
class CrossDomainFramework(Framework):

  def evaluate(self, dataset):
    # TODO: Check if this evaluation has already been done!
    self.notify("Evaluating over '%s'" % dataset.__name__)
    other = Framework(dataset, work_path = self.work_path) #Ensure the store is shared
    other.set_feature_spaces(self.feature_spaces)
    other.set_class_space(self.class_space)

    gs = other.classmap.raw
    classifier = self.classifier
    cl = classifier(other.featuremap.raw)

    md = {}
    md['uuid']='TODO'
    # TODO:  Metadata!!
    result = Result(gs, cl, numpy.arange(gs.shape[0]), md)
    tsr = TaskSetResult([result], md)
    return tsr 

import numpy
import hydrat

from hydrat.frameworks.common import Framework
from hydrat.result.result import Result
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.common.decorators import replace_with_result

import os
from hydrat.display.html import TableSort 
from hydrat.display.summary_fns import sf_featuresets
from hydrat.frameworks.offline import process_results, result_summary_table
from hydrat.result.interpreter import SingleHighestValue, NonZero, SingleLowestValue
# TODO: Allow for feature weighting and selection
# TODO: Produce tasksets, and bring this more in line with the offline framework.
#       This would reduce the burden in implementing weighting and selection.

import logging
logger = logging.getLogger(__name__)

summary_fields=\
  [ ( {'label':"Dataset", 'searchable':True}       , "dataset"       )
  , ( {'label':"Eval", 'searchable':True}       , "eval_dataset"       )
  , ( {'label':"Class Space",'searchable':True}     , "class_space"     )
  , ( {'label':"# Feats",'searchable':True}    , "num_featuresets"    )
  , ( {'label':"Feature Desc",'searchable':True}   , "feature_desc"     )
  , ( {'label':"Learner",'searchable':True}    , "learner"    )
  , ( {'label':"Params",'searchable':True}    , "learner_params"    )
  , ( "Macro-F"       , "macro_fscore"        )
  , ( "Macro-P"     , "macro_precision"     )
  , ( "Macro-R"        , "macro_recall"        )
  , ( "Micro-F"       , "micro_fscore"        )
  , ( "Micro-P"     , "micro_precision"     )
  , ( "Micro-R"        , "micro_recall"        )
  # TODO the next two are missing in the metadata somehow.
  #, ( {'sorter':'digit', 'label':"Learn Time"}    , "avg_learn"     )
  #, ( {'sorter':'digit', 'label':"Classify Time"} , "avg_classify"  )
  , ( {'sorter': None, 'label':"Details"}      , "link"          )
  ]

 
class CrossDomainFramework(Framework):
  # TODO: Refactor with OfflineFramework - there is much in common.
  def __init__(self, dataset, store=None):
    Framework.__init__(self, dataset, store)

    self.interpreter = SingleHighestValue()

  def evaluate(self, dataset):
    # NOTE: This approach to constructing metadata is brittle, in that it will not
    #       reflect changes made to metadata elsewhere.
    md = {}
    md['dataset'] = self.dataset.__name__
    md['class_space'] = self.class_space
    md['feature_desc'] = tuple(sorted(self.feature_spaces))
    md['task_type'] = self.__class__.__name__
    md['eval_dataset'] = dataset.__name__
    md.update(self.learner.metadata)
    if self.store.has_TaskSetResult(md):
      self.notify("Previously evaluated over '%s'" % dataset.__name__)
      return False 
    else:
      self.notify("Evaluating over '%s'" % dataset.__name__)
      other = Framework(dataset, store = self.store) 
      other.set_feature_spaces(self.feature_spaces)
      other.set_class_space(self.class_space)

      classifier = self.classifier
      cl = classifier(other.featuremap.raw)
      md.update(classifier.metadata)

      gs = other.classmap.raw
      result = Result(gs, cl, numpy.arange(gs.shape[0]), md)
      tsr = TaskSetResult([result], md)
      self.store.new_TaskSetResult(tsr)
      return True

  # TODO: Clean up and refactor this cut&paste from offline.
  def generate_output(self, path=None, summary_fn=sf_featuresets, fields = summary_fields):
    """
    Generate HTML output
    """
    if path is None: 
      path = hydrat.config.getpath('paths', 'output')
    if not os.path.exists(path): 
      os.mkdir(path)
    self.notify("Generating output")
    summaries = process_results\
      ( self.store 
      , self.store
      , summary_fn = summary_fn
      , output_path = path
      , interpreter = self.interpreter
      ) 

    # render a HTML version of the summaries
    relevant = list(fields)
    for f_name in self.store.list_FeatureSpaces():
      relevant.append( ({'label':f_name, 'searchable':True}, 'feat_' + f_name) )

    indexpath = os.path.join(path, 'index.html')
    with TableSort(open(indexpath, "w")) as renderer:
      result_summary_table(summaries, renderer, relevant = relevant)
    self.outputP = path

  def upload_output(self, target):
    """
    Copy output to a sepecified destination.
    Useful for transferring results to a webserver
    """
    self.notify("Uploading output to '%s'"% target)
    import updatedir
    updatedir.logger = logger
    updatedir.updatetree(self.outputP, target, overwrite=True)

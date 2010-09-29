import numpy

from hydrat.frameworks.common import Framework
from hydrat.result.result import Result
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.common.decorators import replace_with_result

import os
from hydrat.display.html import TableSort 
from hydrat.display.summary_fns import sf_featuresets
from hydrat.frameworks.offline import process_results, result_summary_table
# TODO: Allow for feature weighting and selection
# TODO: Metadata, results saving, checking for existing result.

import logging
logger = logging.getLogger(__name__)

summary_fields=\
  [ ( {'label':"Dataset", 'searchable':True}       , "dataset"       )
  , ( {'label':"Eval", 'searchable':True}       , "eval_dataset"       )
  , ( {'label':"Class Space",'searchable':True}     , "class_name"     )
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

  def evaluate(self, dataset):
    # TODO: The time taken to read featuremap and classmap metadata by reading in the entire FM and CM 
    #       is nontrivial. This could be refactored to improve performance.
    md = {}
    md.update(self.featuremap.metadata)
    md.update(self.classmap.metadata)
    md.update(self.learner.metadata)
    md['task_type'] = self.__class__.__name__
    md['eval_dataset'] = dataset.__name__
    if self.store.has_TaskSetResult(md):
      self.notify("Previously evaluated over '%s'" % dataset.__name__)
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

  # TODO: Clean up and refactor this cut&paste from offline.
  def generate_output(self, summary_fn=sf_featuresets, fields = summary_fields, interpreter = None):
    """
    Generate HTML output
    """
    self.notify("Generating output")
    summaries = process_results\
      ( self.store 
      , self.store
      , summary_fn = summary_fn
      , output_path= self.outputP
      , interpreter = interpreter
      ) 

    # render a HTML version of the summaries
    relevant = list(fields)
    for f_name in self.store.list_FeatureSpaces():
      relevant.append( ({'label':f_name, 'searchable':True}, 'feat_' + f_name) )

    indexpath = os.path.join(self.outputP, 'index.html')
    with TableSort(open(indexpath, "w")) as renderer:
      result_summary_table(summaries, renderer, relevant = relevant)

  def upload_output(self, target):
    """
    Copy output to a sepecified destination.
    Useful for transferring results to a webserver
    """
    self.notify("Uploading output to '%s'"% target)
    import updatedir
    updatedir.logger = logger
    updatedir.updatetree(self.outputP, target, overwrite=True)
    


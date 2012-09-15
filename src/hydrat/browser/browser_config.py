from hydrat.summary import classification_summary
summary_fn = classification_summary()

from hydrat.result.interpreter import SingleHighestValue 
interpreter = SingleHighestValue()

relevant  = [
  ( {'label':"Dataset", 'searchable':True}       , "dataset"       ),
  ( {'label':"Class Space",'searchable':True}     , "class_space"     ),
  ( {'label':"Feature Desc",'searchable':True}   , "feature_desc"     ),
  ( {'label':"Learner",'searchable':True}    , "learner"    ),
  ( {'label':"Params",'searchable':True}    , "learner_params"    ),
  ( "Macro-F"       , "macro_fscore"        ),
  ( "Macro-P"     , "macro_precision"     ),
  ( "Macro-R"        , "macro_recall"        ),
  ( "Micro-F"       , "micro_fscore"        ),
  ( "Micro-P"     , "micro_precision"     ),
  ( "Micro-R"        , "micro_recall"        ),
  ( {'sorter':'digit', 'label':"Learn Time"}    , "avg_learn"     ),
  ( {'sorter':'digit', 'label':"Classify Time"} , "avg_classify"  ),
]

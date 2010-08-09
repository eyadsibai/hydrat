from hydrat import configuration 
from hydrat.classifier.baseline           import *
from hydrat.classifier.knn                import *
from hydrat.classifier.nearest_prototype  import *
from hydrat.classifier.rainbow            import *
from hydrat.classifier.libsvm             import *
from hydrat.classifier.SVM                import *

__all__ = [ x for x in dir() if x.endswith('L') ]

baseline_learners           = [ randomL, majorityL ]
knn_learners                = [ cosine_1nnL
                              , skew_1nnL
                              , oop_1nnL
                              #, tau_1nnL
                              ]
nearest_prototype_learners  = [ cosine_mean_prototypeL
                              , cosine_gmean_prototypeL
                              , cosine_hmean_prototypeL
                              , skew_mean_prototypeL
                              , skew_gmean_prototypeL
                              , skew_hmean_prototypeL
                              , textcatL
                              #, tau_mean_prototypeL
                              #, tau_gmean_prototypeL
                              #, tau_hmean_prototypeL
                              ]
rainbow_learners            = [ naivebayesL
                              #, prindL
                              #, tfidfL
                              ]
svm_learners                = [ bsvmL, libsvmL ]

all_learners = ( baseline_learners 
               + knn_learners 
               + nearest_prototype_learners
               + rainbow_learners
               #+ svm_learners
               )


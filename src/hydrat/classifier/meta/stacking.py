raise DeprecationWarning, "Needs to be fixed up"
import numpy
from scipy.sparse import csr_matrix
from hydrat.classifier.abstract import Learner, Classifier
from hydrat.preprocessor.model import ImmediateModel
from hydrat.task.taskset import CrossValidate
from hydrat.experiments import Experiment

class StackingLearner(Learner):
  def __init__(self, metalearner, learner_committee, folds = 10, seed = 61383441363):
    #TODO: Better naming
    self.__name__ = '_'.join([ 'stacking'
                             , metalearner.__name__ 
                             ] + [ l.__name__ for l in learner_committee ] 
                            )
    Learner.__init__(self)
    #TODO: Typechecking on metalearner and learner_committee
    self.metalearner= metalearner 
    self.learner_committee= learner_committee 
    self.folds = 10
    self.seed = seed

  def _params(self):
    return dict( folds = self.folds
               , seed = self.seed
               , l0 = self.metalearner.__name__
               , l0_params = self.metalearner.params
               , l1 = [ l.__name__ for l in self.learner_committee ]
               , l1_params = dict((l.__name__, l.params) for l in self.learner_committee)
               )

  def _learn(self, feature_map, class_map):
    train_model = ImmediateModel(feature_map, class_map)
    taskset = CrossValidate(train_model, folds = self.folds, seed = self.seed)
    cl_feats = []
    for learner in self.learner_committee:
      experiment = Experiment(taskset, learner)
      tsr = experiment._run()
      results = tsr.raw_results
      order = numpy.hstack([ r.instance_indices for r in results ]).argsort()
      cl    = numpy.vstack([ r.classifications for r in results])[order]
      cl_feats.append(cl)
    cl_feats = csr_matrix(numpy.hstack(cl_feats))
    metaclassifier = self.metalearner(cl_feats, class_map)
    classif_committee = [ l(feature_map, class_map) for l in self.learner_committee ]
    return StackingClassifier(metaclassifier, classif_committee, self.__name__)
      

class StackingClassifier(Classifier):
  def __init__(self, metaclassifier, classif_committee, name):
    self.__name__ = name
    Classifier.__init__(self)
    self.metaclassifier = metaclassifier
    self.classif_committee = classif_committee 

  def _classify(self, feature_map):
    pred_feats = []
    for c in self.classif_committee:
      pred_feats.append(c(feature_map))
    pred_feats = csr_matrix(numpy.hstack(pred_feats))
    return self.metaclassifier(pred_feats)

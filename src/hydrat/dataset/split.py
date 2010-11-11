from abstract import Dataset
import numpy
import hydrat
from hydrat.preprocessor.model.inducer import map2matrix, matrix2map
from hydrat.task.sampler import stratify, allocate
from hydrat.common.decorators import replace_with_result

class TrainTest(Dataset):
  """
  This class implements a randomized stratified train-test split.
  It can be inherited from as-is if the reproducibility of the split
  is not a concern, or sp_traintest can be used as an example
  of how to use traintest to build an appropriate split
  """
  @replace_with_result
  def sp_traintest(self):
    """
    Return a stratified train-test split, using the first 
    classmap returned by the dataset, and a default train:test
    ratio of 4:1. 
    This is a good example of how to implement your own split
    based on a random train-test allocation.
    Keep in mind that the state of the RNG will affect the
    allocation. Controlling the state of the rng is the resposibility
    of the user. The default implementation uses replace_with_result
    to ensure the split is reported consistently within a run, but
    this split will change if the program is re-run.
    """
    return self.traintest(list(self.classmap_names)[0], 4, hydrat.rng)

  def traintest(self, cm_name, ratio, rng):
    classmap = self.classmap(cm_name)

    # Convert into a matrix representation to facilitate stratification
    ids = classmap.keys()
    matrix = map2matrix(classmap, instance_ids = ids)

    # Stratify and allocate to partitions
    strata_map = stratify(matrix)
    partition_proportions = numpy.array([ratio, 1])
    parts  = allocate( strata_map
                     , partition_proportions
                     , probabilistic = False
                     , rng=rng
                     ) 
    mapping = matrix2map(parts.transpose(), ['train', 'test'], ids)
    return mapping

class CrossValidation(Dataset):
  @replace_with_result
  def sp_crossvalidation(self):
    return self.crossvalidation(list(self.classmap_names)[0], 10, hydrat.rng)

  def crossvalidation(self, cm_name, folds, rng):
    classmap = self.classmap(cm_name)

    # Convert into a matrix representation to facilitate stratification
    ids = classmap.keys()
    matrix = map2matrix(classmap, instance_ids = ids)

    # Stratify and allocate to partitions
    strata_map = stratify(matrix)
    partition_proportions = numpy.array([1] * folds )
    parts  = allocate( strata_map
                     , partition_proportions
                     , probabilistic = False
                     , rng=rng
                     ) 
    parts = numpy.dstack((numpy.logical_not(parts), parts))
    fold_labels = [ 'fold%d' % i for i in xrange(folds) ]
    mapping = matrix2map(parts[:,:,1].transpose(), fold_labels, ids)
    return mapping

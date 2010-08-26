from hydrat.frameworks.preset_split import PresetSplitFramework
from hydrat.common.transform.featureselect import ig_bern_top500
from hydrat.task.transform import transform_taskset

import hydrat.corpora.reuters as reuters

import hydrat.classifier.nearest_prototype as np

from hydrat.corpora.reuters import Reuters21578
from hydrat.dataset.tokenstream import PorterStem
class ReutersSubset(Reuters21578, PorterStem): 
  """
  Subclass of Retuers21578. Implements a reduced split, where only 2000 of the 21578 documents
  are used. This is useful for development work, where we want to quickly test that code is
  working. Inhertiting from PorterStem gives us ts_porterstemmer, which is a tokenstream 
  of porter-stemmed BOW tokens.
  """
  def sp_hydrat_test_subset(self):
    return dict(train=map(str,range(1,1001)),test=map(str,range(1001,2001)),unused=map(str,range(2001,21579)))

if __name__ == "__main__":
  ps = PresetSplitFramework(ReutersSubset())

  # Do classifications in the topics class space
  ps.set_class_space('reuters21578_topics')

  # Use the reduced split defined above
  ps.set_split('hydrat_test_subset')

  # Use cosine-mean nearest prototype. Chosen because it is internal and fast
  ps.set_learner(np.cosine_mean_prototypeL())

  # Use featuremaps in the bag_of_words feature space
  ps.set_feature_spaces('bag_of_words')
  
  # Create a new taskset by carrying out infogain-based top500 feature selection
  ps.transform_taskset(ig_bern_top500)
  ps.run()

  ps.generate_output()

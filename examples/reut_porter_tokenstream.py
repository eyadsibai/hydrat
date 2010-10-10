from hydrat.frameworks.offline import OfflineFramework
from hydrat.common.transform.featureselect import ig_bern_top500
from hydrat.task.transform import transform_taskset

import hydrat.corpora.reuters as reuters

import hydrat.classifier.nearest_prototype as np

from hydrat.corpora.reuters import Reuters21578
from hydrat.dataset.tokenstream import PorterStem
class ReutersSubset(PorterStem, Reuters21578): 
  """
  Subclass of Retuers21578. Implements a reduced split, where only 2000 of the 21578 documents
  are used. This is useful for development work, where we want to quickly test that code is
  working. Inhertiting from PorterStem gives us ts_porterstemmer, which is a tokenstream 
  of porter-stemmed BOW tokens.
  """
  def sp_hydrat_test_subset(self):
    return dict(train=map(str,range(1,1001)),test=map(str,range(1001,2001)),unused=map(str,range(2001,21579)))

from hydrat.common.counter import Counter
def stem_unigram(tokenstream):
  """ Returns the unigram distribution of stems in the TokenStream
  """
  return Counter( token['stem'] for token in tokenstream )

if __name__ == "__main__":
  fw = OfflineFramework(ReutersSubset())

  # Do classifications in the topics class space
  fw.set_class_space('reuters21578_topics')

  # Use the reduced split defined above
  fw.set_split('hydrat_test_subset')

  # Use cosine-mean nearest prototype. Chosen because it is internal and fast
  fw.set_learner(np.cosine_mean_prototypeL())

  # Generate a stem_unigram featureset off the porterstemmer token stream.
  fw.process_tokenstream('porterstemmer', stem_unigram)

  # Use featuremaps in the stem_unigram feature space
  fw.set_feature_spaces('porterstemmer_stem_unigram')
  fw.run()

  # Create a new taskset by carrying out infogain-based top500 feature selection
  # Note that this taskset is saved back into the store.
  fw.transform_taskset(ig_bern_top500)
  fw.run()


# Introduction #

Feature weighting and feature selection are blanket terms used to describe manipulations on a feature space. Since the two manipulations share very many common properties, we refer to them collectively as feature transformations in hydrat.

Conceptually, there are several points at which this could be done:
  1. Modify the input token stream directly
  1. Modify feature vectors as they are fed to a classifier
  1. Modify stored TaskSets

The problem with modifying the input token stream directly is that it can result in information 'leakage'. Some feature weighting/selection approaches (eg Mutual Information) take into account the class distribution of instances. We cannot do this over instances that are intended to be members of the test set, or we will leak information from the test set into the model.

The problem with modifying feature vectors as they are fed to a classifier is that there is no consistent way to save the modified feature vectors, so we have to re-compute them every time. Hydrat supports this mode of operation (TODO details).

The preferred solution of the author is to perform feature selection at the Task level. A Task is a specification of a train/test feature map/class map pair to be fed to a classifier. A TaskSet is simply a grouping of related tasks. For example, in a 10-Fold cross-validation, the TaskSet would contain 10 tasks, where each of the tasks is a 9-1 split of the entire dataset into train/test instances. In the initial cross-validation TaskSet, each instance is duplicated 10 times, once per Task. This may seem wasteful, but it makes sense in the context of feature weighting/selection. After a feature vector is transformed in a class-sensitive fashion (e.g. MI), or even in a context-sensitive fashion (e.g. TFIDF), the feature vector describing a particular instance will be different in each of the 10 folds.

# How to perform Feature Transforms #
Here is an example based on examples/dummy\_split.py

```
  fw = OfflineFramework(dummy.unicode_dummy())
  fw.set_class_space('dummy_default')
  fw.set_feature_space('byte_unigram')
  fw.set_split('dummy_default')
  for l in learners:
    ps.set_learner(l)
    ps.run()
  fw.transform_taskset(TFIDF())
  for l in learners:
    ps.set_learner(l)
    ps.run()
  ps.generate_output()
```

OfflineFramework-derived classes have an method 'transform\_taskset', which takes a transformer as an argument. hydrat handles the generation and capturing of metadata, and the transformed taskset is saved back into the store so that it will not need to be regenerated.

# Implementation of Feature Transforms #
There are two parts to this implementation of feature transforms:
  1. 'Plumbing'
  1. The transform itself

## Plumbing ##
This refers to the function which applies the transform, and handles the metadata. This function is transform\_taskset in [hydrat.task.transform](http://code.google.com/p/hydrat/source/browse/src/hydrat/task/transform.py).

It applies the supplied transformer to each Task in the supplied TaskSet, and updates the feature description of the TaskSet to reflect the transform applied.

A transformed task is generated from the previous task as follows:
```
  transformer.learn(task.train_vectors, task.train_classes)
  t = Task()
  t.train_vectors = transformer.apply(task.train_vectors)
  t.test_vectors  = transformer.apply(task.test_vectors)
  t.train_classes = task.train_classes
  t.test_classes  = task.test_classes
  t.train_indices = task.train_indices
  t.test_indices  = task.test_indices
```


## Transformer API ##

The transformer API is described by the class [Transformer](http://code.google.com/p/hydrat/source/browse/src/hydrat/common/transform/__init__.py).

```
class Transformer(object):
  def learn(feature_map, class_map):
    raise NotImplemented

  def apply(feature_map):
    raise NotImplemented
```

Consider the following implementation of [TFIDF](http://code.google.com/p/hydrat/source/browse/src/hydrat/common/transform/weight.py):
```
class TFIDF(LearnlessTransformer):
  def __init__(self):
    self.__name__ = 'tfidf'
    LearnlessTransformer.__init__(self)

  def apply(self, feature_map):
    weighted_fm = lil_matrix(feature_map.shape, dtype=float)
    instance_sizes = feature_map.sum(axis=1)
    
    # Frequency of each term is the sum alog axis 0
    tf = feature_map.sum(axis=0)
    # Total number of terms in fm
    total_terms = tf.sum()
    
    #IDF for each term
    idf = numpy.zeros(feature_map.shape[1])
    for f in feature_map.nonzero()[1]:
      idf[f] += 1
              
    for i,instance in enumerate(feature_map):
      size = instance_sizes[i]
      # For each term in the instance
      for j in instance.nonzero()[1]: 
        v = feature_map[i, j] 
        term_freq =  float(v) / float(size) #TF        
        weighted_fm[i, j] = term_freq * idf[j] #W_{d,t}
    
    return weighted_fm.tocsr()
```

LearnlessTransformer is a subclass of Transformer that simply does 'pass' on the learn stage. It makes sense in the TFIDF context as TFIDF does not need to learn from class labels.

The apply stage must return a sparse feature map (in the form of a scipy csr sparse matrix). The feature map must have the same number of instances (0-axis), but can have a different number of features(1-axis). This is feature selection. The features can also have different values. This is feature weighting. You can also perform both in a single step if the algorithm suits.

# Further Work #
## Storing Weights ##
One step that occurs in many transforms is a weight calculating step. In may cases, this step can be quite expensive (eg information gain), and it would be advantageous to be able to store it.

From a conceptual perspective, these weights are associated with a given task, and could be stored alongside a task.

From the Store perspective, each task object could be given an additional group node, and then individual nodes can be attached to this, named after the weighting function used.

From the TaskSet perspective, this gets trickier, as we do not want to load every single weight, every single time we use the taskset. The only alternative to this however is a tighter coupling between the TaskSet and the Store. The proposed solution is to extend the interface for accessing Tasks in a store, declaring the set of weights which are required, which would be empty by default.

From the Transformer perspective, getting access to store weights is currently impossible since the interface operates directly over raw featuremaps and classmaps. This is desirable from an extensibility perspective as it minimizes the amount of hydrat-specific work required to implement a Transformer. The downside is that it is not possible for the Transformer to be aware of existing weights. The proposed solution is to modify the transformer interface to promote weights to having first-class status in them. Every transformer has a set of weights, just that in many instances this set is empty. When given a weighting function, the Transformer will check if it already has a set of weights corresponding to that function. If it does, it will use the precomputed weights. If not, it will compute them.

From the Frameworks perspective, the Framework becomes responsible for co-ordinating the loading and saving of weights from the store. When applying a transformer, it will query the transformer for the weights it requires, and attempt to load them from the store. After the transformer is applied, concurrently with saving the derived taskset, it will query the transformer for weights it is aware of, and if any new weights are found it will save them with the corresponding Task.

Unsupervised transforms can be tricky, because in some instances we wish to calculate some parameters that are a function of both the training and test feature maps. For example, in the case of TFIDF, the IDF component could be calculated from the full feature map since it is meant to approximate the universal distribution of term occurrences. However, the transformer interface will not allow this as it only provides a view on the training featuremap. We deal with this by separately learning the global weight vector, then associating it with each of the task nodes. Therefore, when transform\_taskset is applied, the framework will query the store for the weight vector, which has already been precomputed (even though the result for that vector would be different from the result over just the training instances).
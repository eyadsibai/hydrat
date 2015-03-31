# Introduction #

This page is a collection of ideas with no firm implementation schedule. Ideas are specified as assertions ('should', 'must'), but everything is open to discussion.

# Store #

The Store is the low-level core of hydrat. It should be split into two interfaces, an expressive front-end for the rest of hydrat to access and a minimalist backend to make it as easy as possible to implement. The Store functionality itself would then be mapping between these two interfaces. Particular backends could override some aspects of the Store implementation for efficiency reasons. E.G, update can always be expressed as delete, add, but we may choose to override that with a more efficient form of update.

## Forms of store ##

### In-memory ###
  * Pros:
  * Fast
  * Cons:
  * Volatile

### Single-file ###
  * Pros:
  * Easy to share
  * Cons:
  * No native concurrency handling

This is currently the only type of Store we implement.

### Filesystem-based ###
### Database ###

## Caching in Store ##
Caching is a separate issue. This could be handled individually (or completely ignored) by any given implementation of a Store. Alternatively, hydrat could implement its own caches - possibly in the form of a meta-Store, which holds an in-memory store and an on-disk store, and manages them accordingly.

# Interfaces #
Concepts should be partitioned into data and functions. Data interfaces should be specified by an abstract base class, which should use the Store interface for actual data access. The data objects themselves should be lightweight, and mostly concerned with the maintenance of metadata.

Examples of data interfaces:
  * FeatureMap
  * ClassMap
  * TaskSet
  * Result

Function interfaces should be strongly types. How to enforce this is a completely separate issue.

Can we make better use of pickling?

Examples of function interfaces (these examples don't deal with sequencing yet):
  * Learner :: FeatureMap, ClassMap -> Classifier
  * Classifier :: FeatureMap -> Prediction
  * Interpreter :: Prediction -> ClassMap
  * Summary :: ClassMap, ClassMap -> DataPoint
  * Is DataPoint its own class, or just any class?
  * Does Summary need more than a goldstandard and a predicted classmap?
  * TransformLearner :: FeatureMap, ClassMap -> Transformer
  * Transformer :: FeatureMap -> FeatureMap
  * Do we want to make a distinction between supervised and unsupervised transforms?
  * Do we want some way of saving Transformers?

## a generalized learning interface? ##
Learner and TransformLearner look very similar. Is there some way to unify them?

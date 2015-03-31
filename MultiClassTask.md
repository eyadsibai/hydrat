# Introduction #

Multi-class classification tasks refer to classification tasks where one instance may have more than one class labels. While classification models such as Nearest Neighbour (NN) and Naive Bayes (NB) can deal with this situation natively, many other models cannot.

Two popular ways to transform Multi-class tasks to single-class tasks are: creating multi-class categories and binrising the classification tasks.

# Fundamentals #
## Multi-class category ##
The initiative of this method is that we can simply treat _each multi-class_ combination in the training data as _a new single class_. For example, if we have a class set `[A, B, C]` and some training instances:

```
`instance1`: A, B
`instance2`: B, C
`instance3`: C
```


The transformed new class set will be: `[AB, BC, C]`, and the class labels for each training instances will be:

```
`instance1`: AB
`instance2`: BC
`instance3`: C
```

Note that, the number of classification tasks is not changed in this method, but we need to transform the new class set (i.e. `[AB, BC, C]`) back to the original class set (i.e. `[AB, BC, C]`) after the classification.

## Binarisation ##
The idea of this approach is to binarise a multi-class classification task to multiple binary-class classification tasks. Because most models can work with binary-class classification tasks. For instance, if we have a class set `[A, B, C]` and some training instances:

```
`instance1`: A, B
`instance2`: B, C
`instance3`: C
```

After binarisation, we have three class sets: `[A, NotA]`, `[B, NotB]`, `[C, NotC]` and three classification tasks respectively:
```
Task1                Task2               Task3
`instance1`: A       `instance1`: B      `instance1`: NotC
`instance2`: NotA    `instance2`: B      `instance2`: C
`instance3`: NotA    `instance3`: NotB   `instance3`: C
```

Note that, because the classification models integrated into Hydrat have to do each binary classification task separately, the time cost will be increased significantly by using this method. Moreover, we also need to transform the binarised class sets back to the original single class set.

# HowTo in Hydrat #
Here is an example which explains how to deal multi-class classification in Hydrat. This example is based on the example from [GettingStarted](GettingStarted.md).

**Multi-class category**
```
  from hydrat.classifiers.meta.stratified import StratifiedLearner
  cv = CrossValidation(dummy.unicode_dummy())
  cv.set_class_space('dummy_default')
  cv.set_feature_space('byte_unigram')
  for l in learners:
    cv.set_learner(StratifiedLearner(l)) #Simple wrap StratifiedLearner around the base leaner
    cv.run()
  cv.generate_output()
```

**Binarisation**
```
  from hydrat.classifiers.meta.binary import BinaryLearner
  cv = CrossValidation(dummy.unicode_dummy())
  cv.set_class_space('dummy_default')
  cv.set_feature_space('byte_unigram')
  for l in learners:
    cv.set_learner(BinaryLearner(l)) #Simple wrap BinaryLearner around the base leaner
    cv.run()
  cv.generate_output()
```
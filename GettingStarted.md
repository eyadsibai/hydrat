# Introduction #

This page assumes that you have already followed the instructions on the [installation page](Installation.md). It will take you through a basic example of how to use hydrat.

# Downloading the example #

The example can be downloaded [here](http://hydrat.googlecode.com/hg/examples/dummy_singleclass.py).

# Source Code #

The meat of this example is this:

```
  fw = OfflineFramework(unicode_dummy())
  fw.set_class_space('dummy_default')
  fw.set_feature_spaces('byte_unigram')

  def do():
    for l in learners:
      fw.set_learner(l)
      fw.run()

  # Run over crossvalidation split
  fw.set_split('crossvalidation')
  do()

  # Run over train/test split
  fw.set_split('traintest')
  do()

  # Perform tfidf weighting
  from hydrat.common.transform.weight import TFIDF
  fw.transform_taskset(TFIDF())
  do()

  # Extend the task with an additional feature space
  fw.extend_taskset('codepoint_unigram')
  do()
```

## Initializing the framework ##
```
  fw = OfflineFramework(unicode_dummy())
```

This line configures an instance of OfflineFramework using the unicode\_dummy() dataset.

## Setting a class space ##
```
  fw.set_class_space('dummy_default')
```

This line sets the class space we will work in.

## Setting a feature space ##
```
  fw.set_feature_spaces('byte_unigram')
```

This line sets the feature space we will work in.
Note that more than one feature space can be set. To do so, provide a list of space names to set\_feature\_spaces, and hydrat will take care of producing the concatenation of the spaces.

## Setting a split ##
```
  fw.set_split('crossvalidation')
```

This line sets the split. In this case, we are using an automatically-generated cross-validation split. This split was declared in the following snippet of code:

```
from hydrat.dataset.split import TrainTest, CrossValidation
class unicode_dummy(dummy.unicode_dummy, TrainTest, CrossValidation): pass
```

By subclassing CrossValidation, the unicode\_dummy class received an automatically-generated random stratified 10-fold cross-validation split. You can control the auto-generation behaviour, as well as program the entire split manually.

## Setting a learner ##
```
    fw.set_learner(l)
```

This line sets the learner we will work with. Note that in this example, a list of learners is specified earlier, and we iterate over each of them, setting them and running them in turn.

## Running an experiment ##
```
    fw.run()
```

This line tells the framework to run the learner over the generated task, and produce output.

## Generating output ##
```
  fw.generate_output()
```

This line tells the framework to generate a HTML rendering of the result of our cross-validation.

# Running the example #
Programs using hydrat are just normal python programs, and are run just like any other python program. For this example, we would run it as follows:

```
python dummy_singleclass.py
```

# Examining the output #
Assuming you are using a default configuration of hydrat, output will have been produced in `./work/OfflineFramework/dummy-unicode100/output`.

You can open `./work/OfflineFramework/dummy-unicode100/output/index.html` in any recent webbrowser to view the results of the cross-validation.
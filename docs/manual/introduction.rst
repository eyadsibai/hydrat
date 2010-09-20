===============
What is hydrat?
===============

hydrat is a framework for comparing classification algorithms. In itself, this is nothing new.
Packages such as weka have been providing this functionality for a long time. Hydrat is special
for several reasons:

- hydrat wraps existing packages rather than reimplementing algorithsm. We have a weka wrapper!
	Rather than reinvent the wheel, hydrat provides a straightforward abstraction of how a classifier
	should behave, and provides implementations to map this behaviour onto a large variety of external
	classification packages. You can also easily add your own classifiers.

- hydrat provides features for managing the entire experimental workflow. Rather than specify a data format, we specify
	a data interface. This approach is much more flexible, because it allows arbitrary formats (such as ARFF) to be easily ported to hydrat.
	It also allows easy access to data from disparate sources, leveraging Python's vast array of libraries. For example, consider a text 
	classification task. The documents are stored as individual text files, but the category labels are stored in a MySQL database. You just 
	need to write a small amount of Python code which implements a class that meets hydrat's data model, and you can quickly start working 
	with this data.

- hydrat provides a framework for managing features. Raw data needs to be processed into a certain representation in order for anything
	useful to be done with it. hydrat helps to automate repetitive aspects of this process. 

- hydrat provides a framework for managing class labels. This means that it is easy to do cross-dataset comparisons. For example, given two
	language identification datasets, it is possible to learn a model using data from one dataset and apply it directly to data from another 
	dataset, with no more effort required than just dealing with each dataset in isolation.

- hydrat stores intermediate results. This means that by sacrificing disk space, you can save yourself future compute time. For example,
	you can store unigram counts for a given dataset. If you return later, and wish to repeat and experiment with say, a different classifier, 
	or a different feature selection algorithm, you can save yourself from having to recompute the unigram count.

- hydrat is commited to reproducibility in scientific computing. By letting hydrat manage the intermediate stages between raw data and final 
	results, you ensure that your experiments can be exactly replicated by anybody else, simply by providing them the raw data and the code 
	which specifies the experiments.

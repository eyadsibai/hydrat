# Introduction #

Hydrat accesses raw data via a programmatic, declarative interface, in contrast to a conventional flat-file approach via data formats such as ARFF. The advantage of accessing data this way is that it allows the user to integrate data from multiple sources, as well as from different storage systems, such as:

  * Flat files
  * Databases
  * Data Models (e.g forum\_features TODO:link)


# Details #

The dataset API allows the user to inform hydrat of the following classes of information:

## TokenStreams ##

Tokenstreams are methods which begin with the 'ts' prefix. They should be a mapping from instance identifiers to a sequence of tokens representing that instance. POS-tagged output is an example of such a token stream: each instance is represented by a list of dictionaries, each representing a particular token.

### Special Tokenstreams ###
These tokenstreams are only special by convention. No checks are run for whether they have been abused.

ts\_text

## Feature Maps ##

Feature maps are methods which begin with the 'fm' prefix. They should be a mapping from instance identifiers to a mapping from features to values. An example of this is a bag-of-words for each instance, although this may be better expressed as a counting process on a ts\_text stream.

## Class Maps ##

Class maps are methods which begin with the 'cm' prefix. They should be a mapapping from instance identifiers to a list of classes which the instance belongs to. The classes should be identified by a unique string. For example, in a langid dataset an instance may be as follows:

> 'document1.txt': ['en','de']

Be careful to wrap the class of single-class instances in a list. A common bug is forgetting to do this, and ending up with a multi-class instance which is a member of a set of classes each bearing a single-letter name! (TODO: Add a warning in hydrat for this, and perhaps a flag to disallow strings from declaring sets of classes)

## Splits ##

Splits are methods which begin with the 'sp' prefix. They should be a mapping from a partition identifier to a list of instance identifiers present in that partition. This can be used for declaring things such as train/test splits, as well as cross-validation splits. Note that the naming of splits is used in some parts of hydrat to identify the type of task.

## Sequences ##

Sequences are methods which begin with the 'sq' prefix. They should return a list of lists of instance identifiers, where each inner list represents a logical group, and the order within the list should reflect the order of the instances within that group. For example, a dataset representing a forum would return a list of lists, where each inner list represents a thread, and the instance represents a particular post. Another example is a dataset for part-of-speech tagging, where each inner list represents a sentence and each member of the list is a particular word.

# TODO #
  * Examples
  * Implementation advice
# Introduction #

One of the key facilities hydrat provides is the ability to work with features of any sort. Many times, these features are derived as counts over a bag-of-tokens model of a document, for example the endemic bag-of-words model. Hydrat provides facilities for dealing with standard tokenizations via the datasets.text module. Dataset instances can derive from datasets.text.Text and implement a text() method, which returns a mapping from instance identifiers to the fulltext of the instance. Any dataset which derives a subclass of datasets.text.Text will automatically be mapped into the feature space provided by that class. For example, deriving from dataset.text.ByteUnigram adds a fm\_byte\_unigram method to the Dataset instance, which in turn reads the instance's text() method, and performs the required tokenization.

# Why we need TokenStream #
Currently, the implementation of methods such as fm\_byte\_unigram lump token generation and distribution counting into a single step. This is reasonably sensible in the simplistic view of a token as a string. However, in many NLP techniques it is useful to think of the token as a more structured entity. POS taggers for example are able to annotate tokens. [GENIA](http://www-tsujii.is.s.u-tokyo.ac.jp/GENIA/tagger/) is such an example, and it produces structured tokens with the following fields:

```
word1   base1   POStag1 chunktag1 NEtag1
word2   base2   POStag2 chunktag2 NEtag2
  :       :        :       :        :
```

There is additional processing that can be done over such tokens. For example, we may want to count all the occurences of a particlar type of Named Entity, rather than the entity itself. Thus, we may wish to store the tokens that we generate before we process them further into distribution counts. TokenStream is the intermediate representation of a processed token stream, which can then be used to build feature maps.

# What tokenstream must does #
  * Serializes to an on-disk format
  * Stored in a PyTables VLArray, using an [http://www.pytables.org/docs/manual/ch04.html#AtomClassDescr](ObjectAtom.md) (which effectively serializes as a pickled string)
  * Exists as a sub-node of specific datasets, at the same level as "feature\_data" and "class\_data"
  * Tokens remain browseable in ViTables because ViTables automatically unpacks the pickled objects.
  * Each row in the VLArray is a (pickled) list of dictionaries

# TODO #
  * Provide metadata facilities, including metadata lookup. Currently, generation is controlled by a simple decision based on whether the dataset,tokenizer pair already exists.

# Implementation Details #

At the in-memory level, TokenStreams are lists of dictionaries. Each dictionary represents a structured token. For example a token produced by the GENIA tagger looks like this:

```
{'POStag': 'VBZ', 'chunktag': 'O', 'base': 'Be', 'word': 'Is', 'NEtag': 'O', start:0, end:3}
```

Additions were made to various APIs:
  * We introduce the 'ts' prefix to the DatasetAPI. This prefix denotes methods that take no arguments, and return a mapping from instance identifiers to TokenStreams (i.e., lists of dictionaries representing structured tokens). The tokenstream (ts) portion of the DatasetAPI is consistent with the implementation of feature maps(fm), class maps(cm), class spaces(cs) and splits (sp).
  * hydrat.preprocessor.model.inducer.dataset.DatasetInducer.process\_Dataset gained a new optional argument 'tss', representing 'TokenStreams', whereby the names of tokenstreams to be processed from the dataset should be specified.
  * The StoreAPI gained add\_TokenStream and get\_TokenStream to handle the serialization of tokenstreams into the hdf5 backend used by the Store.


# Relationship to the DatasetAPI #

The present design of the DatasetAPI is such that a dataset exports feature maps in a fashion that does not need to know anything about the StoreAPI. In order to maintain this decoupling, we sacrifice the guarantee that all of a Dataset's feature maps are declared together with the dataset itself. Instead, we defer generation of feature maps derived from TokenStreams to a later step in the pipeline.

# Generating features from TokenStreams #

A new convenience method was added to hydrat.frameworks.Framework: process\_tokenstream. This method accepts two arguments: the name of the tokenstream and a processor function. The name of the tokenstream is used to retrieve the tokenstream from the Store instance associated with the Framework. The tokenstream will be automatically induced from the Dataset instance associated with the Framework if it is not yet present. The processor function must be a function that takes a single argument (the tokenstream, in list-of-dictionaries format) and returns a single mapping from feature to value. The simplest token processors are counters; they count the number of occurrences of a particular token (or token feature). For example, for tokens that represent words this becomes the standard bag-of-words model. For tokens that contain information such as Named Entity type, we could count all the occurrences of a particular NE type, such as 'protein', or 'place name', or 'linux-package-name' etc. The name of the featuremap thus generated will be the name attribute of the processor function. This allows the processor function to be implemented as a callable class if some sort of parametrization is required.
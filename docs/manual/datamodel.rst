=================
hydrat Data Model
=================

Instead of specifying a data format, hydrat specifies an interface for accessing data.
hydrat is focused on instance level-classification, but what an instance represents is
entirely up to the user. In conventional text-categorization research and instance will 
usually represent a single document, but hydrat has also been used in research involving data
from online forums, where an instance can be a post in the forum for example.

In hydrat, a Dataset is a Python class that extends hydrat.dataset.Dataset. 
It should provide at least one method whose name starts with 'cm_', and one method
that starts with 'fm_'. Methods whose names start with 'fm_' denote feature maps. They
should take no arguments and should return a dictionary mapping from an instance identifier
to a dictionary mapping from a feature name to a count for that feature. Methods whose names
start with 'cm_' denote class maps. They should take no arguments, and should return a dictionary
from an instance identifier to a list of class labels to which that instance belongs. Note that
the set of instances is never explicitly specified - it is up to the user to ensure that all 'fm_' 
methods and all 'cm_' methods in a class return a dictionary with the same set of keys. hydrat 
will perform some sanity checks when dealing with data, and mismatched instance identifier sets
will generally raise an exception.

The dataset interface allows for a third type of method, which starts with 'cs_' and specifies a class
space. This is useful for specifying class spaces which are common to multiple datasets, for example a
class space representing a set of languages. It is possible that several datasets should map into this space,
but that not all languages are specified in all the datasets. For example, one dataset may cover only 10 european
languages, whereas another may cover 5 asian languages and 5 european languages. A 'cs_' method should
take no arguments and should return a list of possible class labels.

hydrat was originally designed to research performance in text classification. To this end, several helper classes
have been defined to facilitate this. :class:`hydrat.dataset.text.TextDataset` extends :class:`hydrat.dataset.Dataset`,
and declares a 'text' method which should be overridden by a user-defined class. The text method should take no arguments,
and should return a dictionary mapping from an instance identifier to the raw text of that instance. This approach
allows hydrat to abstract away the details of implementing tokenization, by providing convenient mixins to handle this.
For example, if a user wishes to experiment with byte unigram and bigram models, they should declare a class which
extends :class:`hydrat.dataset.text.ByteUnigram` and :class:`hydrat.dataset.text.ByteBigram`, and implements a 
'text()' method. By doing this, hydrat will automatically tokenize the text using unigram and bigram tokenizers, and
provide the methods 'fm_byte_unigram' and 'fm_byte_bigram' respectively.

One of the conventional approaches to modelling text is the so-called bag-of-words model, which is simply a 
count of the frequency of each word in a document, discarding and contextual information about the words. 
However, we cannot simply tokenize raw data. Raw data must use some encoding. Rather than assume this is always
ascii, hydrat provides another set of helper classes which extend :class:`hydrat.dataset.encoded.EncodedTextDataset`, which 
in turn extends :class:`hydrat.dataset.text.TextDataset`. Classes which extend :class:`EncodedTextDataset` 
should provide one additional method, 'encodings', which should return a dictionary mapping from an instance identifier to
the encoding to use in decoding its text. Two mixins which extend :class:`EncodedTextDataset` are provided. The first
is :class:`UTF8`, which simply declares that every document is utf-8 encoded. The second is `AutoEncoding`, and is more 
interesting. It uses the :mod:`chardet` module to automatically guess the encoding of a given document.

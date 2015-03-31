

# Introduction #

http://www.comp.mq.edu.au/programming/

Macquarie University and the Australasian Language Technology Association are organising a programming competition for university undergraduate and Masters students.

The Language Technology Programming Competition is formatted as a "shared task": all participants compete to salve the same problem. The problem highlights an active area of research and programming in the area of language technology. You can easily obtain good results with little effort yet nobody has managed to obtain 100% correct results so
far.

# Installing hydrat #
Follow the instructions given in [Installation](Installation.md).

# Setting up for the competition #
## Installing external classifiers ##
hydrat's configuration system is set up to search for external tools on the user's $PATH. It is up to the user to install these tools. Amongst the supported are:

  * [libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
  * [bsvm](http://www.csie.ntu.edu.tw/~cjlin/bsvm/)
  * [maxent](http://homepages.inf.ed.ac.uk/lzhang10/maxent_toolkit.html)
  * [weka](http://www.cs.waikato.ac.nz/ml/weka/) (slow)
  * [nltk](http://www.nltk.org/) (slow)

## Obtaining data ##
Download the training/development data package:
```
wget http://web.science.mq.edu.au/~diego/pub/altw2010.tgz
```

Unpack the dataset. This will create a folder 'dataset'
```
tar xvzf altw2010.tgz
```

hydrat searches for datasets on the basis of directory names. Therefore, we need to rename the dataset directory to what hydrat will search for.
```
mv dataset altw2010-langid
```

Tell hydrat to search for tools/data and write a configuration file. This will create a .hydratrc file in your current directory.
```
hydrat configure
```

## Pre-generated feature spaces ##

A .h5 file containing pre-tokenized datasets is available. WARNING: 400M, expands to 1.2GB.
```
wget http://hum.csse.unimelb.edu.au/~mlui/langid/resources/langid.tgz
```

Datasets indexed:
  * TwitterWikipedia: synthetically generated tweet-sized messages from wikipedia
  * twitter: small set of twitter messages annotated as eng/non-eng
  * Wikipedia: From NAACL2010 paper
  * TCL: From NAACL2010 paper
  * EuroGOV: From NAACL2010 paper
  * ALTW2010: The a/m dataset for this task

Unique tokens across all datasets:
|codepoint\_unigram  | 10479   |
|:-------------------|:--------|
|codepoint\_bigram   | 414713  |
|codepoint\_trigram  | 1642856 |
|byte\_unigram	     | 226     |
|byte\_bigram	     | 28008   |
|byte\_trigram	     | 431324  |

There is also a .h5 file containing only the pre-tokenized ALTW2010 dataset. 199M, expands to 299M
```
wget http://hum.csse.unimelb.edu.au/~mlui/langid/resources/altw2010-langid.tgz
```


## Run a sample experiment ##
Download the sample experiment:
```
wget http://hydrat.googlecode.com/hg/examples/langid/altw2010-langid.py
```

Run it (you will need libsvm installed):
```
python altw2010-langid.py
```

### Details of the sample experiment ###
This experiment runs the following classifiers over the train/test split of the data in the byte/codepoint uni/bi/trigram spaces:

  * np.skew\_mean\_prototypeL()
  * knn.skew\_1nnL()
  * baseline.majorityL()
  * svm.libsvmExtL(kernel\_type='linear')
  * svm.libsvmExtL(kernel\_type='rbf')

It will take some time to complete! If you wish to run over all of the features/spaces, you might want to leave it to run overnight. Otherwise, consider pruning the set of feature/classifier combinations to run over. A good place to start may be with with just byte-bigrams with a skew\_mean\_prototype learner. Keep in mind that skew mean prototype generates one prototype per class, so you will only ever get a single-class output from it.

When initializing a framework, you can choose where the .h5 file created should be stored by specifying the path as follows:

```
fw = OfflineFramework(dataset, store='path/to/store')
```

If you don't specify `store`, hydrat will create one in the current working directory, using the name of the outermost calling file.
## Viewing results ##

### Static Generation ###
This generates a set of static html pages which can be uploaded to another server. This method is invoked by calling `fw.generate_output('path/to/outputdir')`, which causes the framework to generate output in the named directory. These pages can be viewed directly in a webbrowser.

### hydrat browser ###
hydrat includes a browser for its Store. This browser can be invoked via `hydrat browse <.h5 file>`. It is implemented as a [cherrypy](http://www.cherrypy.org/) application, and gives you more insight into the underlying data.

### ViTables ###
[ViTables](http://vitables.berlios.de/download/index.html) is a generic viewer for hdf5 files. Hydrat's Store uses hdf5 for storing data, so any hydrat .h5 file can be viewed in ViTables. To install ViTables on ubuntu, you will need python-qt4 installed. You can then install from sources via the package's setup.py .
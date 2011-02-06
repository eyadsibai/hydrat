# Online classification framework.
# TODO: Refactor out the commonalities with Framework.
# TODO: Support feature selection and weighting

import os
import logging
import time
import numpy
import scipy.sparse

import hydrat
import hydrat.common.extractors as ext
from hydrat.store import Store, StoreError, NoData, AlreadyHaveData
from hydrat.preprocessor.model.inducer.dataset import DatasetInducer
from hydrat.common import as_set
from hydrat.preprocessor.features.transform import union
from hydrat.preprocessor.model.inducer import invert_text
from hydrat.common.pb import ProgressIter
from hydrat.frameworks.common import Framework

class OnlineFramework(Framework):
  def __init__( self
              , dataset
              , store = None
              ):
    Framework.__init__(self, dataset, store)

    self.vectorize = None
    self.__classifier = None

  def notify(self, str):
    self.logger.info(str)

  def is_configurable(self):
    return self.feature_spaces is not None\
       and self.class_space is not None\
       and self.learner is not None\
       and self.interpreter is not None

  def configure(self):
    if self.is_configurable():
      if not all(f.startswith('byte') for f in self.feature_spaces):
        raise ValueError, "can only handle byte-derived feature spaces (for now!)"

      # Set up feature extraction machinery for incoming text.
      fns = {}
      fis = {}
      exs = {}
      for feature_space in self.feature_spaces:
        feat_names = self.store.get_Space(feature_space)
        fns[feature_space] = feat_names
        fis[feature_space] = dict( (k,v) for v, k in enumerate(feat_names) )
        # TODO: Improve the sophistication of this.
        #       In particular, need to be able to work with extractors 
        #       declared by the user.
        #       This is also where we should handle tokenstream transforms
        exs[feature_space] = getattr(ext, feature_space.split('_')[1])

      # TODO: May be able to refactor this against the inducer
      def vectorize(text):
        batch_size = len(text)
        fms = []
        for feature_space in sorted(self.feature_spaces):
          feat_names = fns[feature_space]
          feat_index = fis[feature_space] 
          extractor  = exs[feature_space]

          feat_map = scipy.sparse.dok_matrix((batch_size, len(feat_names)), dtype='uint64')
          for i, t in enumerate(text):
            inv_text = extractor(t)
            for feat in inv_text:
              if feat in feat_index:
                j = feat_index[feat]
                feat_map[i,j] = inv_text[feat]
          fms.append( feat_map )
        return scipy.sparse.hstack(fms).tocsr()
      
      self.vectorize = vectorize
      self.__classifier = self.classifier

      
  # TODO: split off a separate batch_classify method, so we can feed this into the same
  #       framework that is used to support external langid tasks.
  def classify(self, text):
    # classify a text instance
    # need to generate the corresponding features first
    solo = False # Was only one instance supplied to classify?
    if isinstance(text, str):
      text = [text]
      solo = True
    elif isinstance(text, unicode):
      raise ValueError, "Can only handle byte streams for now"
    feat_map = self.vectorize(text)
    cl = self.__classifier(feat_map)
    result = self.interpreter(cl)
    classlabels = numpy.array(self.classlabels)
    if solo:
      return classlabels[result[0]]
    else:
      return [classlabels[r] for r in result]

  @property
  def metadata(self):
    # TODO: Note hydrat version?
    retval = dict\
      ( feature_spaces = sorted(self.feature_spaces)
      , class_spaces   = self.class_space
      , learner        = {'name':self.learner.__name__, 'params':self.learner._params()}
      , split          = self.split_name
      , interpreter    = self.interpreter.__name__
      , dataset        = self.dataset.__name__
      , instance_space = self.dataset.instance_space
      )
    return retval

  def serve_xmlrpc(self, host='localhost', port=9000):
    # TODO: Allow configuration of classify server via XMLRPC.
    from SimpleXMLRPCServer import SimpleXMLRPCServer
    import json
    server = SimpleXMLRPCServer((host, port), logRequests=True, encoding='ascii')
    server.register_introspection_functions()

    def configuration():
      self.logger.info('Serving configuration')
      return json.dumps(self.metadata)
      
    def classify(text):
      if isinstance(text, str):
        self.logger.info('Classifying 1 instance of len %d', len(text))
      else: 
        self.logger.info('Classifying %d instances', len(text))
      response = {}

      start = time.time()
      text = [ t.encode('utf8') for t in text ]
      outcome = self.classify(text)
      
      response['timeTaken'] = time.time() - start
      response['prediction'] = [ list(x) for x in outcome ]
      # TODO: Add confidence?
      return json.dumps(response)

    server.register_function(configuration)
    server.register_function(classify)

    try:
      print 'Use Crtl+C to exit'
      server.serve_forever()
    except KeyboardInterrupt:
      print 'Exiting'

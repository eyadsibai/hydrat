import logging
import numpy
from hydrat import config
from hydrat.store import NoData, AlreadyHaveData
from hydrat.preprocessor.model.inducer import map2matrix
from hydrat.common.pb import ProgressIter
from hydrat.common import as_set
from hydrat.common.sequence import sequence2matrix

logger = logging.getLogger(__name__)

class DatasetInducer(object):

  def __init__(self, store):
    self.store = store

  def process_Dataset(self, dataset, fms=None, cms=None, tss=None, sqs=None):
    logger.debug('Processing %s', dataset.__name__)
    logger.debug('  fms: %s', fms)
    logger.debug('  cms: %s', cms)
    logger.debug('  tss: %s', tss)
    logger.debug('  sqs: %s', sqs)

    dsname = dataset.__name__

    # Work out if this is the first time we encounter this dataset
    if hasattr(self.store.datasets, dsname):
      logger.debug("Already had dataset '%s'", dsname)
    else:
      logger.debug("Adding new dataset '%s'", dsname)
      self.store.add_Dataset(dsname, dataset.instance_space, dataset.instance_ids)

    fms = as_set(fms)
    cms = as_set(cms)
    tss = as_set(tss)
    sqs = as_set(sqs)

    present_fm = set(self.store.list_FeatureSpaces(dsname))
    present_cm = set(self.store.list_ClassSpaces(dsname))
    present_ts = set(self.store.list_TokenStreams(dsname))
    present_sq = set(self.store.list_Sequence(dsname))

    logger.debug("present_fm: %s", str(present_fm))
    logger.debug("present_cm: %s", str(present_cm))
    logger.debug("present_ts: %s", str(present_ts))
    logger.debug("present_sq: %s", str(present_sq))

    logger.debug("For dataset '%s', Store has:", dsname)
    logger.debug("  %d Feature Maps", len(present_fm))
    logger.debug("  %d Class Maps", len(present_cm))
    logger.debug("  %d Token Streams", len(present_ts))
    logger.debug("  %d Sequences", len(present_sq))

    # Handle explicit class spaces 
    for key in set(dataset.classspace_names):
      logger.debug("Processing explicit class space '%s'", key)
      try:
        c_metadata = {'type':'class','name':key}
        self.store.add_Space(dataset.classspace(key), c_metadata)
      except AlreadyHaveData, e:
        logger.debug(e)

    # Handle all the class maps
    for key in cms - present_cm:
      logger.debug("Processing class map '%s'", key)
      try:
        self.add_Classmap(dsname, key, dataset.classmap(key))
      except AlreadyHaveData,e :
        logger.debug(e)

    # Handle all the feature maps
    for key in fms - present_fm:
      logger.debug("Processing feature map '%s'", key)
      try:
        self.add_Featuremap(dsname, key, dataset.featuremap(key))
      except AlreadyHaveData,e :
        logger.warning(e)
        # TODO: Why are we calling pdb for this?
        import pdb;pdb.post_mortem()
      except AttributeError,e :
        logger.warning(e)
        # TODO: Make ignoring this error configurable

    # Handle all the token streams
    for key in tss - present_ts:
      logger.debug("Processing token stream '%s'", key)

      try:
        self.add_TokenStreams(dsname, key, dataset.tokenstream(key))
      except AlreadyHaveData,e :
        logger.warning(e)

    # Handle all the sequences
    for key in sqs - present_sq:
      logger.debug("Processing sequence '%s'", key)
      self.add_Sequence(dsname, key, dataset.sequence(key))


  def add_Sequence(self, dsname, seq_name, sequence):
    # This involves converting the sequence representation from lists of identifers 
    # in-dataset identifiers to a matrix. 
    # Axis 0 represents the parent and axis 1 represents the child.
    # A True value indicates a directed edge from parent to child.
    instance_ids = self.store.get_Space(dsname)
    index = dict((k,i) for i,k in enumerate(instance_ids))
    sqlist = [ [index[id] for id in s] for s in sequence ]
    sqmatrix = sequence2matrix(sqlist) 
    logger.debug("Adding Sequence'%s' to Dataset '%s'", seq_name, dsname)
    self.store.add_Sequence(dsname, seq_name, sqmatrix)
    #TODO: Attach metadata to the sequence node - is there any we actually want?

  def add_TokenStreams(self, dsname, stream_name, tokenstreams):
    metadata = dict()
    instance_ids = self.store.get_Space(dsname)

    tslist = [tokenstreams[i] for i in instance_ids]
    logger.debug("Adding Token Stream '%s' to Dataset '%s'", stream_name, dsname)
    self.store.add_TokenStreams(dsname, stream_name, tslist)

    #TODO: Attach metadata to the tokenstream node
      

  
  def add_Featuremap(self, dsname, space_name, feat_dict):
    metadata = {'type':'feature','name':space_name}

    # One pass to compute the full set of features
    logger.debug("Computing Feature Set")
    feat_labels = reduce\
                    ( set.union
                    , (    set(f.keys()) 
                      for  f 
                      in   ProgressIter(feat_dict.values(),"Computing Feature Set")
                      )
                    )
    logger.debug("  Identified %d unique features", len(feat_labels))

    # Handle the feature space information
    try:
      space = self.store.get_Space(space_name)
      logger.debug("Extending a previous space")
      new_labels = list(feat_labels - set(space))
      new_labels.sort()
      feat_labels = space
      feat_labels.extend(new_labels)
      space_tag = self.store.extend_Space(space_name, feat_labels)
    except NoData:
      feat_labels = list(feat_labels)
      feat_labels.sort()
      logger.debug("Creating a new space")
      self.store.add_Space(feat_labels, metadata)

    instance_ids = self.store.get_Space(dsname)
    assert set(instance_ids) == set(feat_dict.keys())

    n_inst = len(feat_dict)
    n_feat = len(feat_labels)

    logger.debug("Computing feature map")
    # Build a list of triplets:
    # (instance#, feat#, value)
    feat_map = []
    feat_index = dict( (k,v) for v, k in enumerate(feat_labels) )
    for i, id in enumerate(ProgressIter(instance_ids,label='Computing Feature Map')):
      for feat in feat_dict[id]:
        j = feat_index[feat]
        feat_map.append((i,j,feat_dict[id][feat]))

    logger.debug("Adding map to store")
    self.store.add_FeatureDict(dsname, space_name, feat_map)

  def add_Classmap(self, dsname, space_name, docclassmap):
    if not config.getboolean('debug','allow_str_classset'):
      if any(isinstance(d, str) or isinstance(d, unicode) for d in docclassmap.values()):
        raise ValueError, "str detected as classset - did you forget to wrap classmap values in a list?"
      
    classlabels = reduce(set.union, (set(d) for d in docclassmap.values()))
    c_metadata = {'type':'class','name':space_name}
    try:
      c_space = self.store.get_Space(space_name)
      #TODO: Being a subset of a stored space is OK!!! - has this been sorted?
      if not classlabels <= set(c_space):
        raise ValueError, "Superfluous classes: %s" % (classlabels - set(c_space))
      # Replace with the stored one, as ordering is important
      classlabels = c_space
    except NoData:
      classlabels = sorted(classlabels)
      self.store.add_Space(classlabels, c_metadata)

    if self.store.has_Data(dsname, space_name):
      raise ValueError, "Already have data for dataset '%s' in space '%s'"% (dsname, space_name)

    instance_ids = self.store.get_Space(dsname)
    class_map = map2matrix(docclassmap, instance_ids, classlabels)
    self.store.add_ClassMap(dsname, space_name, class_map)


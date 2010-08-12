import logging
import numpy
from hydrat.store import NoData, AlreadyHaveData
from hydrat.preprocessor.model.inducer import class_matrix 

logger = logging.getLogger(__name__)

class DatasetInducer(object):

  def __init__(self, store):
    self.store = store

  def process_Dataset(self, dataset, fms=None, cms=None):
    logger.debug('Processing %s', dataset.__name__)
    logger.debug('  fms: %s', fms)
    logger.debug('  cms: %s', cms)
    dsname = dataset.__name__

    # Work out if this is the first time we encounter this dataset
    try:
      ds_tag = self.store.resolve_Dataset(dsname)
      logger.debug("Already had dataset '%s'", dsname)
    except NoData:
      logger.debug("Adding new dataset '%s'", dsname)
      ds_tag = self.store.add_Dataset(dataset.instance_ids, dsname)

    # Work out which feature maps and/or class maps we have been asked to process
    def as_set(s):
      if s is None: return set()
      if isinstance(s, str): return set([s])
      else: return set(s)

    fms = as_set(fms)
    cms = as_set(cms)

    present_fm = set(self.store.list_FeatureSpaces(dsname))
    present_cm = set(self.store.list_ClassSpaces(dsname))

    logger.debug("present_fm: %s", str(present_fm))
    logger.debug("present_cm: %s", str(present_cm))

    logger.debug("Store has %d Feature Maps and %d Class Maps for dataset '%s'", len(present_fm), len(present_cm), dsname)
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
        self.add_Classmap(ds_tag, key, dataset.classmap(key))
      except AlreadyHaveData,e :
        logger.debug(e)

    # Handle all the feature maps
    for key in fms - present_fm:
      logger.debug("Processing feature map '%s'", key)

      try:
        self.add_Featuremap(ds_tag, key, dataset.featuremap(key))
      except AlreadyHaveData,e :
        logger.warning(e)
        # TODO: Why are we calling pdb for this?
        import pdb;pdb.post_mortem()
  
  def add_Featuremap(self, ds_tag, name, feat_dict):
    metadata = {'type':'feature','name':name}

    # One pass to compute the full set of features
    logger.debug("Computing Feature Set")
    feat_labels = reduce(set.union, (set(f.keys()) for f in feat_dict.values()))
    logger.debug("  Identified %d unique features", len(feat_labels))

    # Handle the feature space information
    try:
      space_tag = self.store.resolve_Space(metadata)
      logger.debug("Extending a previous space")
      space = self.store.get_Space(space_tag)
      new_labels = list(feat_labels - set(space))
      new_labels.sort()
      feat_labels = space
      feat_labels.extend(new_labels)
      space_tag = self.store.extend_Space(feat_labels, space_tag)
    except NoData:
      feat_labels = list(feat_labels)
      feat_labels.sort()
      logger.debug("Creating a new space")
      space_tag = self.store.add_Space(feat_labels, metadata)

    instance_ids = self.store.instance_identifiers(ds_tag)
    assert set(instance_ids) == set(feat_dict.keys())

    n_inst = len(feat_dict)
    n_feat = len(feat_labels)

    logger.debug("Computing feature map")
    # Build a list of triplets:
    # (instance#, feat#, value)
    feat_map = []
    feat_index = dict( (k,v) for v, k in enumerate(feat_labels) )
    for i, id in enumerate(instance_ids):
      for feat in feat_dict[id]:
        j = feat_index[feat]
        feat_map.append((i,j,feat_dict[id][feat]))

    logger.debug("Adding map to store")
    self.store.add_FeatureDict(ds_tag, space_tag, feat_map)

  def add_Classmap(self, ds_tag, name, docclassmap):
    classlabels = reduce(set.union, (set(d) for d in docclassmap.values()))
    c_metadata = {'type':'class','name':name}
    try:
      class_tag = self.store.resolve_Space(c_metadata)
      c_space = self.store.get_Space(class_tag)
      #TODO: Being a subset of a stored space is OK!!!
      if not classlabels <= set(c_space):
        raise ValueError, "Superfluous classes: %s" % (classlabels - set(c_space))
      # Replace with the stored one, as ordering is important
      classlabels = c_space
    except NoData:
      classlabels = sorted(classlabels)
      class_tag = self.store.add_Space(classlabels, c_metadata)

    if self.store.has_Data(ds_tag, class_tag):
      class_name = self.store.get_Metadata(class_tag)['name']
      raise ValueError, "Already have data for dataset '%s' in space '%s'"% (ds_tag, class_name)

    instance_ids = self.store.instance_identifiers(ds_tag)
    class_map = class_matrix(docclassmap, instance_ids, classlabels)
    self.store.add_ClassMap(ds_tag, class_tag, class_map)


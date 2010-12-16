import uuid
import time
import tables
import os
import warnings
import logging
import numpy
import datetime as dt
warnings.simplefilter("ignore", tables.NaturalNameWarning)
from scipy.sparse import lil_matrix, coo_matrix

from hydrat import config
from hydrat.common import progress
from hydrat.common.metadata import metadata_matches, get_metadata
from hydrat.preprocessor.model import ClassMap
from hydrat.preprocessor.features import FeatureMap
from hydrat.result.result import Result
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.task.task import Task
from hydrat.task.taskset import TaskSet
from hydrat.common.pb import ProgressIter

from hydrat.common.decorators import deprecated

logger = logging.getLogger(__name__)

class StoreError(Exception): pass
class NoData(StoreError): pass
class AlreadyHaveData(StoreError): pass
class InsufficientMetadata(StoreError): pass

# TODO: Provide a facility for saving splits
# TODO: Avoid leaking uuids out with tasksets and/or results.

# Features are internally stored as sparse arrays, which are serialized at the
# pytables level to tables of instance, feature, value triplets. We support
# both Integer and Real features.

class IntFeature(tables.IsDescription):
  ax0    = tables.UInt64Col()
  ax1    = tables.UInt64Col()
  value  = tables.UInt64Col()

class RealFeature(tables.IsDescription):
  ax0    = tables.UInt64Col()
  ax1    = tables.UInt64Col()
  value  = tables.Float64Col()

class BoolFeature(tables.IsDescription):
  ax0    = tables.UInt64Col()
  ax1    = tables.UInt64Col()
  value  = tables.BoolCol()
# TODO: Declare a configurable compression filter
#         tables.Filters(complevel=5, complib='zlib') 

STORE_VERSION = 3

def update_h5store(fileh):
  """
  Update the format of a h5file-backed store from an earlier version.
  This is done as an incremental procress, i.e. an update 0->2 is done as
  0->1->2
  The fileh is assumed to be writeable.
  An update from a file without version number is taken to mean that is is either
  version unknown or a brand new file.
  """
  logger.debug("Running update_h5store")
  root = fileh.root

  version = root._v_attrs['version'] if 'version' in root._v_attrs else 0
  
  if version < 1:
    # No version, or new file
    # Ensure that the major nodes exist
    logger.debug('updating to version 1')
    for node in ['spaces', 'datasets', 'tasksets', 'results']:
      if not hasattr(root, node):
        fileh.createGroup( root, node )
    # Check that the dataset nodes are well-formed
    for dsnode in root.datasets:
      if not hasattr(dsnode, 'tokenstreams'):
        logger.debug('Node %s did not have tokenstreams node; adding.', dsnode._v_name)
        fileh.createGroup( dsnode, "tokenstreams" )
      if not hasattr(dsnode, 'sequence'):
        logger.debug('Node %s did not have sequence node; adding.', dsnode._v_name)
        fileh.createGroup( dsnode, "sequence" )
  if version < 2:
    # In version 2, we introduce the concept of instance spaces, detaching the instance
    # identifiers from the dataset nodes and instead attaching them to the space nodes
    logger.debug('updating to version 2')
    for dsnode in root.datasets:
      # Move the instance id node to spaces
      id_node = dsnode.instance_id
      id_node._v_attrs['date']      = dt.datetime.now().isoformat() 
      id_node._v_attrs['size']      = len(dsnode.instance_id)
      id_node._v_attrs['type']      = 'instance'
      id_node._v_attrs['name']      = dsnode._v_name
      id_node._v_attrs['encoding']  = 'utf8' # to be safe, in case we had e.g. utf8 filenames
      fileh.moveNode(dsnode.instance_id, root.spaces, dsnode._v_name)
      # Unless otherwise specified, the instance space is the dataset name
      dsnode._v_attrs['instance_space'] = dsnode._v_name

    # Add the instance space metadata to all tasksets
    for tsnode in root.tasksets:
      tsnode._v_attrs.instance_space = tsnode._v_attrs.dataset
      for t in tsnode:
        t._v_attrs.instance_space = t._v_attrs.dataset
        
    # Add the instance space metadata to all results
    for rnode in root.results:
      rnode._v_attrs.instance_space = rnode._v_attrs.dataset
      if hasattr(rnode._v_attrs, 'eval_dataset'):
        rnode._v_attrs.eval_space = rnode._v_attrs.eval_dataset
      for node in rnode:
        if node._v_name == 'summary':
          for summary in node:
            summary._v_attrs.instance_space = summary._v_attrs.dataset
            if hasattr(summary._v_attrs, 'eval_dataset'):
              summary._v_attrs.eval_space = summary._v_attrs.eval_dataset
        else:
          node._v_attrs.instance_space = node._v_attrs.dataset
          if hasattr(node._v_attrs, 'eval_dataset'):
            node._v_attrs.eval_space = node._v_attrs.eval_dataset
  if version < 3:
    # In version 3, we add weights associated with task nodes
    for tsnode in root.tasksets:
      for t in tsnode:
        fileh.createGroup(t, 'weights')
        

  logger.debug("updated store from version %d to %d", version, STORE_VERSION)
  root._v_attrs['version'] = STORE_VERSION
  fileh.flush()

class Store(object):
  """
  This is the master store class for hydrat. It manages all of the movement of data
  to and from disk.
  """
  def __init__(self, path, mode='r'):
    """
    The store object has four major nodes:
    # spaces
    # datasets
    # tasksets
    # results
    """
    self.path = path
    logger.debug("Opening Store at '%s', mode '%s'", self.path, mode)
    self.fileh = tables.openFile(self.path, mode=mode)
    self.mode = mode

    try:
      self._check_writeable()
      update_h5store(self.fileh)
    except IOError:
      pass

    self.root = self.fileh.root
    self.datasets = self.root.datasets
    self.spaces = self.root.spaces
    self.tasksets = self.root.tasksets
    self.results = self.root.results

    if not 'version' in self.root._v_attrs or self.root._v_attrs['version'] != STORE_VERSION:
      raise ValueError, "Store format is outdated; please open the store as writeable to automatically update it"

  
  def __str__(self):
    return "<Store mode '%s' @ '%s'>" % (self.mode, self.path)

  def __del__(self):
    self.fileh.close()

  ###
  # Utility Methods
  ###
  def _check_writeable(self):
    if self.mode not in "wa":
      raise IOError, "Store is not writeable!"

  def _read_sparse_node(self, node, shape=None):
    """
    We allow the shape to be overloaded so that we can accomodate
    feature maps where the underlying feature space has grown
    """
    dtype = node._v_attrs.dtype
    if shape is None: shape = node._v_attrs.shape
    ax0 = node.read(field='ax0')
    ax1 = node.read(field='ax1')
    values = node.read(field='value')
    m = coo_matrix((values,numpy.vstack((ax0,ax1))), shape=shape)
    m = m.tocsr()
    return m

  def _add_sparse_node( self
                      , where
                      , name
                      , type
                      , data
                      , title=''
                      , filters=None
                      ):
    node = self.fileh.createTable( where 
                                 , name
                                 , type
                                 , title
                                 , filters
                                 , expectedrows = data.nnz
                                 )
    attrs = node._v_attrs
    setattr(attrs, 'dtype', data.dtype)
    setattr(attrs, 'shape', data.shape)
    # Add the features to the table
    inst, feat = data.nonzero()
    node.append(numpy.rec.fromarrays((inst.astype('uint64'), feat.astype('uint64'), data.data)))
    self.fileh.flush()
  
  ###
  # Add
  ###
  def add_Space(self, labels, metadata):
    """
    Add a space to the store. metadata must contain at
    least 'name' and 'type'.
    """
    self._check_writeable()
    try:
      self.resolve_Space(metadata)
      raise AlreadyHaveData, "Already have space %s" % metadata
    except NoData:
      pass
    if 'type' not in metadata:
      raise InsufficientMetadata, "metadata must contain type"
    if 'name' not in metadata:
      raise InsufficientMetadata, "metadata must contain name"

    logger.debug( "Adding a %s space '%s' of %d Features"
                    , metadata['type']
                    , metadata['name']
                    , len(labels)
                    )

    # TODO: Maybe check the metadata for encoding?
    # Weak assumption that if the first label is unicode,
    # they all are unicode
    if isinstance(labels[0], unicode):
      encoding = 'utf8'
      # We encode all labels because PyTables rejects unicode
      labels = [ l.encode(encoding) for l in labels ]
    else:
      encoding = 'ascii'

    metadata['encoding'] = encoding
      
    new_space = self.fileh.createArray( self.fileh.root.spaces
                                      , metadata['name']
                                      , labels
                                      )
    new_space.attrs.date = dt.datetime.now().isoformat() 
    new_space.attrs.size = len(labels)
    for key in metadata:
      setattr(new_space.attrs, key, metadata[key])
    assert metadata_matches(new_space._v_attrs, metadata)

  def extend_Space(self, space_name, labels):
    # We do this by checking that the new space is a superset of the
    # old space, with the labels in the same order, then we delete the old 
    # space and add a new space.
    space = self.get_Space(space_name)
    if any( old != new for old, new in zip(space, labels)):
      raise StoreError, "New labels are not an extension of old labels"

    if len(labels) < len(space):
      raise StoreError, "New labels are less than old labels"

    if len(labels) == len(space):
      logger.debug("Space has not changed, no need to extend")

    logger.debug("Extending '%s' from %d to %d features", space_name, len(space), len(labels))
    metadata = self.get_SpaceMetadata(space_name)

    encoding = metadata['encoding']
    if encoding != 'ascii':
      labels = [l.encode(encoding) for l in labels]

    # Delete the old node
    self.fileh.removeNode( getattr(self.spaces, space_name) )
    # Create the new node
    new_space = self.fileh.createArray( self.fileh.root.spaces
                                      , space_name
                                      , labels
                                      )
    # Transfer the metadata
    for key in metadata:
      setattr(new_space.attrs, key, metadata[key])

    # Update the size
    new_space.attrs.size = len(labels)
    # Update the date
    new_space.attrs.date = dt.datetime.now().isoformat() 
    self.fileh.flush()


  def add_Dataset(self, name, instance_space, instance_ids):
    self._check_writeable()
    if hasattr(self.datasets, name):
      raise AlreadyHaveData, "Already have dataset by name %s", name

    # Create a group for the DataSet
    ds = self.fileh.createGroup( self.fileh.root.datasets
                               , name
                               )

    attrs = ds._v_attrs

    # Note down our metadata
    attrs.name              = name
    attrs.instance_space    = instance_space
    attrs.date              = dt.datetime.now().isoformat()
    attrs.num_instances     = len(instance_ids)

    # Create the instance_id array
    if hasattr(self.spaces, instance_space):
      # Check that the spaces match
      space = self.get_Space(instance_space)
      if (space != numpy.array(instance_ids)).any():
        raise ValueError, "Instance identifiers don't match existing instance space"
    else:
      self.add_Space(instance_ids, {'type':'instance', 'name':instance_space})

    # Create a group for Feature Data 
    self.fileh.createGroup( ds
                          , "feature_data"
                          , "Feature Data"
                          )

    # Create a group for Class Data 
    self.fileh.createGroup( ds
                          , "class_data"
                          , "Class Data"
                          )

    # Create a group for Token Streams
    self.fileh.createGroup( ds
                          , "tokenstreams"
                          , "Token Streams"
                          )

    # Create a group for Token Streams
    self.fileh.createGroup( ds
                          , "sequence"
                          , "Instance sequencing information"
                          )

  def add_FeatureDict(self, dsname, space_name, feat_map):
    self._check_writeable()
    logger.debug("Adding feature map to dataset '%s' in space '%s'", dsname, space_name)
    ds = getattr(self.datasets, dsname)
    space = getattr(self.spaces, space_name)

    group = self.fileh.createGroup( ds.feature_data
                                  , space._v_name
                                  , "Sparse Feature Map %s" % space._v_name
                                  )
    group._v_attrs.date  = dt.datetime.now().isoformat()
    group._v_attrs.type  = 'int' if all(isinstance(i[2], int) for i in feat_map) else 'float'

    fm_node = self.fileh.createTable( group
                                    , 'feature_map'
                                    , IntFeature if group._v_attrs.type == 'int' else RealFeature 
                                    , title = 'Sparse Feature Map'
                                    , filters = tables.Filters(complevel=5, complib='zlib') 
                                    , expectedrows = len(feat_map)
                                    )

    # Initialize space to store instance sizes.
    n_inst = len(self.get_Space(dsname))
    # TODO: instance sizes makes no sense in the context of float features!
    #       the whole notion of instance size should disappear, as it only makes sense for count features

    instance_sizes = numpy.zeros(n_inst, dtype='uint64')
    
    attrs = fm_node._v_attrs
    setattr(attrs, 'dtype', group._v_attrs.type)
    setattr(attrs, 'shape', (n_inst, len(space)))

    # Add the features to the table
    feature = fm_node.row
    for i, j, v in feat_map:
      feature['ax0'] = i
      feature['ax1'] = j
      feature['value'] = v
      feature.append()
      if group._v_attrs.type == 'int':
        instance_sizes[i] += v # Keep tally of instance size

    # Store the instance sizes
    fm_sizes = self.fileh.createArray( group
                                     , 'instance_size'
                                     , instance_sizes
                                     , title = 'Instance Sizes'
                                     )

    self.fileh.flush()

  def add_FeatureMap(self, dsname, space_name, feat_map):
    self._check_writeable()
    logger.debug("Adding feature map to dataset '%s' in space '%s'", dsname, space_name)
    ds = getattr(self.datasets, dsname)
    space = getattr(self.spaces, space_name)

    group = self.fileh.createGroup( ds.feature_data
                                  , space._v_name
                                  , "Sparse Feature Map %s" % space._v_name
                                  )
    group._v_attrs.date  = dt.datetime.now().isoformat()
    group._v_attrs.type  = int if issubclass(feat_map.dtype.type, numpy.int) else float

    # Initialize space to store instance sizes.
    instance_sizes = numpy.array(feat_map.sum(axis=1))[0]

    _typ = IntFeature if issubclass(feat_map.dtype.type,numpy.int) else RealFeature 
    fm_node = self._add_sparse_node\
                  ( group
                  , 'feature_map'
                  , _typ 
                  , feat_map
                  , filters = tables.Filters(complevel=5, complib='zlib') 
                  )
     
    # Store the instance sizes
    fm_sizes = self.fileh.createArray( group
                                     , 'instance_size'
                                     , instance_sizes
                                     , title = 'Instance Sizes'
                                     )

    self.fileh.flush()


  def add_ClassMap(self, dsname, space_name, class_map):
    self._check_writeable()
    logger.debug("Adding Class Map to dataset '%s' in space '%s'", dsname, space_name)
    ds = getattr(self.datasets, dsname)
    space = getattr(self.spaces, space_name)

    num_inst = self.get_SpaceMetadata(ds._v_attrs['instance_space'])['size'] 
    num_classes = len(space)

    # Check that the map is of the right shape for the dataset and space
    if class_map.shape != (num_inst, num_classes):
      raise StoreError, "Wrong shape for class map!"

    group =  self.fileh.createGroup( ds.class_data
                                   , space._v_name
                                   , "Data for %s" % space._v_name
                                   )
    group._v_attrs.date = dt.datetime.now().isoformat()
    group._v_attrs.uuid = uuid.uuid4() # Not currently used anywhere

    cm_node = self.fileh.createCArray( group
                                     , 'class_map'
                                     , tables.BoolAtom()
                                     , class_map.shape
                                     , title = 'Class Map'
                                     , filters = tables.Filters(complevel=5, complib='zlib') 
                                     )
    cm_node[:] = class_map
    cm_node.flush()
                               
  def add_TaskSet(self, taskset):
    # TODO: Find some way to make this atomic! Otherwise, we can write incomplete tasksets!!
    self._check_writeable()
    taskset_uuid = taskset.metadata['uuid']

    if 'date' not in taskset.metadata:
      taskset.metadata['date'] = dt.datetime.now().isoformat()

    taskset_entry_tag = str(taskset_uuid)
    taskset_entry = self.fileh.createGroup(self.tasksets, taskset_entry_tag)
    taskset_entry_attrs = taskset_entry._v_attrs

    for key in taskset.metadata:
      setattr(taskset_entry_attrs, key, taskset.metadata[key])

    logger.debug('Adding a taskset %s', str(taskset.metadata))

    for i,task in enumerate(ProgressIter(taskset.tasks, label="Adding Tasks")):
      self._add_Task(task, taskset_entry)
    self.fileh.flush()

    return taskset_entry_tag

  ###
  # Resolve
  ###
  #TODO: Are these ever used?
  def resolve_Space(self, desired_metadata):
    """
    Uniquely resolve a single space
    """
    spaces = self.resolve_Spaces(desired_metadata)
    if spaces == []:
      raise NoData, "No space matching given metadata"
    elif len(spaces) > 1:
      raise InsufficientMetadata, "%d spaces matching given metadata" % len(spaces)
    else:
      return spaces[0]

  def resolve_Spaces(self, desired_metadata):
    """
    Linear search for matching spaces
    @param desired_metadata: key-value pairs that the desired space must satisfy
    @type desired_metadata: dict
    @return: tags corresponding to spaces with the desired metadata
    @rtype: list of strings
    """
    names = []
    for space in self.spaces:
      if metadata_matches(space.attrs, desired_metadata):
        names.append(space._v_name)

    return names 

  ###
  # List
  ###
  def list_ClassSpaces(self, dsname = None):
    if dsname is None:
      return set(s._v_name for s in self.spaces if s._v_attrs.type == 'class')
    else:
      ds = getattr(self.datasets, dsname)
      return set(node._v_name for node in ds.class_data)

  def list_FeatureSpaces(self, dsname = None):
    if dsname is None:
      return set(s._v_name for s in self.spaces if s._v_attrs.type == 'feature')
    else:
      ds = getattr(self.datasets, dsname)
      return set(node._v_name for node in ds.feature_data)

  def list_InstanceSpaces(self):
    return set(s._v_name for s in self.spaces if s._v_attrs.type == 'instance')

  def list_Datasets(self):
    return set( ds._v_attrs.name for ds in self.datasets)

  ###
  # Get
  ###
  def get_SpaceMetadata(self, space_name):
    """
    @param space_name: Identifier of the relevant space
    @type tag: string
    @rtype: dict of metadata key-value pairs
    """
    space = getattr(self.spaces, space_name)
    metadata = dict(   (key, getattr(space._v_attrs,key)) 
                  for  key 
                  in   space._v_attrs._v_attrnamesuser
                  )
    return metadata

  def get_DatasetMetadata(self, dsname):
    """
    @param dsname: Identifier of the relevant dataset
    @type dsname: string
    @rtype: dict of metadata key-value pairs
    """
    ds = getattr(self.datasets, dsname)
    metadata = dict(   (key, getattr(ds._v_attrs,key)) 
                  for  key 
                  in   ds._v_attrs._v_attrnamesuser
                  )
    return metadata

  def get_Space(self, space_name):
    """
    @param space_name: Name of the space 
    @rtype: pytables array
    """
    try:
      space = getattr(self.spaces, space_name)
    except tables.exceptions.NoSuchNodeError:
      raise NoData, "Store does not have space '%s'" % space_name
    metadata = self.get_SpaceMetadata(space_name)
    data = space.read()
    try:
      encoding = metadata['encoding']
    except KeyError:
      logger.warning('Space %s does not have encoding data!', tag)
      encoding = 'nil'
    if encoding != 'nil' and encoding != 'ascii':
      data = [ d.decode(encoding) for d in data ]
    return data

  # Deprecated upon introduction of instance spaces. use get_Space instead.
  @deprecated
  def get_InstanceIds(self, dsname):
    """
    @param dsname: Dataset name
    @return: Instance identifiers for dataset
    @rtype: List of strings
    """
    return self.get_Space(dsname)

  def has_Data(self, dsname, space_name):
    try:
      ds = getattr(self.datasets, dsname)
    except tables.exceptions.NoSuchNodeError:
      return False
    return (  hasattr(ds.class_data,   space_name) 
           or hasattr(ds.feature_data, space_name)
           )

  def get_ClassMap(self, dsname, space_name):
    """
    @param dsname: Name of the dataset
    @param space_name: Name of the class space
    @return: data corresponding to the given dataset in the given class space
    @rtype: pytables array
    """
    ds = getattr(self.datasets, dsname)
    try:
      class_node = getattr(ds.class_data, space_name) 
      data = getattr(class_node, 'class_map')
    except AttributeError:
      raise NoData

    metadata = dict\
                 ( dataset=dsname
                 , class_space=space_name
                 , instance_space=ds._v_attrs.instance_space
                 )
    return ClassMap(data.read(), metadata)

  def get_FeatureMap(self, dsname, space_name):
    """
    @param dsname: Name of the dataset
    @param space_name: Name of the feature space
    @return: data corresponding to the given dataset in the given feature space
    @rtype: varies 
    """
    ds = getattr(self.datasets, dsname)
    space = self.get_Space(space_name)
    try:
      feature_node = getattr(ds.feature_data, space_name)
    except AttributeError:
      raise NoData

    data_type = feature_node._v_attrs.type
    logger.debug("Returning matrix of type %s", data_type)
    fm = getattr(feature_node, 'feature_map')
    n_inst = self.get_SpaceMetadata(ds._v_attrs['instance_space'])['size'] 
    n_feat = len(space)
    m = self._read_sparse_node(fm,shape=(n_inst, n_feat))
    metadata = dict\
                 ( dataset=dsname
                 , feature_desc=(space_name,)
                 , instance_space=ds._v_attrs.instance_space
                 )
    return FeatureMap(m, metadata) 

  @deprecated
  def get_SizeData(self, dsname, space_name):
    """
    TODO: Generalize this to getInstanceWeightData, which is a vector we can use
    to weight instances in a given feature space. For token-based spaces, this 
    represents the number of tokens in the space.

    @param dsname: Name of the dataset
    @param space_name: Name of the feature space
    @return: data corresponding to the size of each instance in this space
    @rtype: pytables array
    """
    ds = getattr(self.datasets, dsname)
    try:
      feature_node = getattr(ds.feature_data, space_name) 
      data = getattr(feature_node, 'instance_size')
    except AttributeError:
      raise NoData
    return data.read()

  # Deprecate get_data, since it doesn't make as much sense now that we have 
  # removed a level of indirection by eliminating uuids for spaces
  @deprecated
  def get_Data(self, dsname, space_metadata):
    """
    Convenience method for data access which compiles relevant metadata
    as well.
    """
    s_type = space_metadata['type']
    s_name = space_metadata['name']

    if s_type == 'class':
      data = self.get_ClassMap(dsname, s_name)
    elif s_type == 'feature':
      data = self.get_FeatureMap(dsname, s_name)
    else:
      raise StoreError, "Unknown data type: %s" % s_type

    logger.debug("Retrieved %s data for '%s' in space '%s'", s_type, dsname, s_name)
    return data 


  def new_TaskSet(self, taskset):
    """Convenience method which checks no previous taskset has matching 
    metadata. Useful for situations where we build up the tasksets incrementally
    """
    if self.has_TaskSet(taskset.metadata):
      raise AlreadyHaveData, "Already have taskset %s" % str(taskset.metadata)
    if 'uuid' in taskset.metadata:
      logger.warning('new taskset should not have uuid!')
    taskset_uuid = uuid.uuid4()
    taskset.metadata['uuid'] = taskset_uuid
    try:
      self.add_TaskSet(taskset)
    except tables.exceptions.NodeError:
      raise AlreadyHaveData, "Node already exists in store!"
      

  def _add_Task(self, task, ts_entry): 
    self._check_writeable()
    task_uuid = uuid.uuid4()
    task.metadata['uuid'] = task_uuid
    task_tag = str(task_uuid)

    # Create a group for the task
    task_entry = self.fileh.createGroup(ts_entry, task_tag)
    task_entry_attrs = task_entry._v_attrs

    # Add the metadata
    for key in task.metadata:
      setattr(task_entry_attrs, key, task.metadata[key])

    # Add the class matrices 
    # TODO: Current implementation has the side effect of expanding all tasks,
    #       meaning we lose the memory savings of an InMemoryTask. Maybe this
    #       is not a big deal, but if it is we need to look into how to handle
    #       it.
    self.fileh.createArray(task_entry, 'train_classes', task.train_classes)
    self.fileh.createArray(task_entry, 'test_classes', task.test_classes)
    self.fileh.createArray(task_entry, 'train_indices', task.train_indices)
    self.fileh.createArray(task_entry, 'test_indices', task.test_indices)
    tr = task.train_vectors
    te = task.test_vectors

    self._add_sparse_node( task_entry
                         , 'train_vectors'
                         , IntFeature if issubclass(tr.dtype.type,numpy.integer) else RealFeature 
                         , tr
                         , filters = tables.Filters(complevel=5, complib='zlib') 
                         )
    self._add_sparse_node( task_entry
                         , 'test_vectors'
                         , IntFeature if issubclass(te.dtype.type,numpy.integer) else RealFeature 
                         , te 
                         , filters = tables.Filters(complevel=5, complib='zlib') 
                         )

    sqr = task.train_sequence
    sqe = task.test_sequence

    if sqr is not None:
      self._add_sparse_node( task_entry
                           , 'train_sequence'
                           , BoolFeature
                           , sqr
                           , filters = tables.Filters(complevel=5, complib='zlib') 
                           )

    if sqe is not None:
      self._add_sparse_node( task_entry
                           , 'test_sequence'
                           , BoolFeature
                           , sqe
                           , filters = tables.Filters(complevel=5, complib='zlib') 
                           )
    weights_node = self.fileh.createGroup(task_entry, 'weights')
    for key in task.weights:
      new_weight = self.fileh.createArray\
                      ( weights_node
                      , key
                      , task.weights[key]
                      )
      new_weight.attrs.date = dt.datetime.now().isoformat() 

    self.fileh.flush()

  def extend_Weights(self, taskset):
    # TODO: Do we need to perform a check for some kind of characteristic
    #       of the weight?
    taskset_entry  = getattr(self.tasksets, str(taskset.metadata['uuid']))
    for task in taskset.tasks:
      task_tag = str(task.metadata['uuid'])
      task_entry     = getattr(taskset_entry, task_tag)
      for key in task.weights:
        if not hasattr(task_entry.weights, key):
          new_weight = self.fileh.createArray\
                          ( task_entry.weights
                          , key
                          , task.weights[key]
                          )
          new_weight.attrs.date = dt.datetime.now().isoformat() 
    self.fileh.flush()
                
  def has_TaskSet(self, desired_metadata):
    """ Check if any taskset matches the specified metadata """
    return bool(self._resolve_TaskSet(desired_metadata))

  def get_TaskSet(self, desired_metadata, weights=None):
    """ Convenience function to bypass tag resolution """
    tags = self._resolve_TaskSet(desired_metadata)
    if len(tags) == 0: raise NoData
    elif len(tags) > 1: raise InsufficientMetadata
    try:
      return self._get_TaskSet(tags[0],weights=weights)
    except tables.NoSuchNodeError:
      logger.warning('Removing damaged TaskSet node with metadata %s', str(desired_metadata))
      self.fileh.removeNode(self.tasksets, tags[0], recursive=True)
      raise NoData

  def _get_TaskSetMetadata(self, taskset_tag):
    try:
      taskset_entry  = getattr(self.tasksets, taskset_tag)
    except AttributeError:
      raise KeyError, str(taskset_tag)
    metadata = get_metadata(taskset_entry)
    return metadata

  def _del_TaskSet(self, taskset_tag):
    if not hasattr(self.tasksets, taskset_tag):
      raise KeyError, str(taskset_tag)
    self.fileh.removeNode(self.tasksets, taskset_tag, True)

  def _get_TaskSet(self, taskset_tag, weights = None):
    try:
      taskset_entry  = getattr(self.tasksets, taskset_tag)
    except AttributeError:
      raise KeyError, str(taskset_tag)
    metadata = get_metadata(taskset_entry)
    tasks = []
    for task_entry in taskset_entry._v_groups.values():
      tasks.append(self._get_Task(task_entry, weights=weights))
    tasks.sort(key=lambda r:r.metadata['index'])
    return TaskSet(tasks, metadata)

  def _get_Task(self,task_entry, weights=None): 
    metadata = get_metadata(task_entry)
    t = Task()
    t.metadata = metadata
    t.train_classes  = task_entry.train_classes.read()
    t.train_indices  = task_entry.train_indices.read()
    t.test_classes   = task_entry.test_classes.read()
    t.test_indices   = task_entry.test_indices.read()
    t.train_vectors  = self._read_sparse_node(task_entry.train_vectors)
    t.test_vectors   = self._read_sparse_node(task_entry.test_vectors)
    if hasattr(task_entry, 'train_sequence'):
      t.train_sequence = self._read_sparse_node(task_entry.train_sequence)
    else:
      t.train_sequence = None
    if hasattr(task_entry, 'test_sequence'):
      t.test_sequence = self._read_sparse_node(task_entry.test_sequence)
    else:
      t.test_sequence  = None
    t.weights = {}
    if weights is not None:
      for key in weights:
        if key in task_entry.weights:
          t.weights[key] = getattr(task_entry.weights, key).read()
        else:
          t.weights[key] = None
    return t

  def _resolve_TaskSet(self, desired_metadata):
    """Returns all tags whose entries match the supplied metadata"""
    desired_keys = []
    for node in self.tasksets._v_groups:
      attrs = getattr(self.tasksets,node)._v_attrs
      if metadata_matches(attrs, desired_metadata):
        desired_keys.append(node)
    return desired_keys

  def new_TaskSetResult(self, tsr):
    """Convenience method which checks no previous TaskSetResult has matching metadata"""
    if self.has_TaskSetResult(tsr.metadata):
      raise AlreadyHaveData, "Already have tsr %s" % str(tsr.metadata)
    if 'uuid' in tsr.metadata:
      logger.warning('new tsr should not have uuid!')
    tsr.metadata['uuid'] = uuid.uuid4()
    self.add_TaskSetResult(tsr)

  def has_TaskSetResult(self, desired_metadata):
    """ Check if the TaskSetResult already exists """
    return bool(self._resolve_TaskSetResults(desired_metadata))

  def get_TaskSetResult(self, desired_metadata):
    """ Convenience function to bypass tag resolution """
    tags = self._resolve_TaskSetResults(desired_metadata)
    if len(tags) == 0: raise NoData
    elif len(tags) > 1: raise InsufficientMetadata
    return self._get_TaskSetResult(tags[0])

  def _get_TaskSetResultMetadata(self, tsr_tag):
    tsr_entry  = getattr(self.results, tsr_tag)
    return get_metadata(tsr_entry)

  def _del_TaskSetResult(self, tsr_tag):
    if not hasattr(self.results, tsr_tag):
      raise KeyError, str(tsr_tag)
    self.fileh.removeNode(self.results, tsr_tag, True)


  def _get_TaskSetResult(self, tsr_tag):
    try:
      tsr_entry  = getattr(self.results, tsr_tag)
    except AttributeError:
      raise KeyError, str(tsr_tag)
    metadata = get_metadata(tsr_entry)
    results = []
    for node in tsr_entry._v_groups.values():
      if node._v_name != 'summary':
        results.append(self._get_Result(node))

    try:
      results.sort(key=lambda r:r.metadata['index'])
    except KeyError:
      logger.warning("Tasks do not have index- returning in unspecified order")
      
    tsr = TaskSetResult(results, metadata)
    if 'instance_space' in metadata:
      try:
        tsr.instance_space = self.get_Space(metadata['instance_space'])
      except NoData:
        pass
    if 'class_space' in metadata:
      try:
        tsr.class_space = self.get_Space(metadata['class_space'])
      except NoData:
        pass
    return tsr

  def _get_Result(self, result_entry):
    metadata = get_metadata(result_entry)
    goldstandard     = result_entry.goldstandard.read()
    classifications  = result_entry.classifications.read()
    instance_indices = result_entry.instance_indices.read()
    r = Result( goldstandard
              , classifications
              , instance_indices
              , metadata
              )
    return r

  def _resolve_TaskSetResults(self, desired_metadata):
    """Returns all tags whose entries match the supplied metadata"""
    keys = self.results._v_groups.keys()
    desired_keys = []
    for key in keys:
      attrs = getattr(self.results,key)._v_attrs
      if metadata_matches(attrs, desired_metadata):
        desired_keys.append(key)
    return desired_keys

  def add_TaskSetResult(self, tsr, additional_metadata={}):
    self._check_writeable()
    try:
      tsr_uuid = tsr.metadata['uuid']
    except KeyError:
      tsr_uuid = uuid.uuid4()
      tsr.metadata['uuid'] = tsr_uuid

    if 'date' not in tsr.metadata:
      tsr.metadata['date'] = dt.datetime.now().isoformat()

    tsr_entry_tag = str(tsr_uuid)
    tsr_entry = self.fileh.createGroup(self.results, tsr_entry_tag)
    tsr_entry_attrs = tsr_entry._v_attrs

    for key in tsr.metadata:
      setattr(tsr_entry_attrs, key, tsr.metadata[key])
    for key in additional_metadata:
      setattr(tsr_entry_attrs, key, additional_metadata[key])

    for i,result in enumerate(tsr.raw_results):
      self._add_Result(result, tsr_entry, dict(index=i))
    self.fileh.flush()
    return tsr_entry_tag

  def _add_Result(self, result, tsr_entry, additional_metadata={}):
    self._check_writeable()
    try:
      result_uuid = result.metadata['uuid']
    except KeyError:
      result_uuid = uuid.uuid4()
      result.metadata['uuid'] = result_uuid
    result_tag = str(result_uuid)

    # Create a group for the result
    result_entry = self.fileh.createGroup(tsr_entry, result_tag)
    result_entry_attrs = result_entry._v_attrs

    # Add the metadata
    for key in result.metadata:
      setattr(result_entry_attrs, key, result.metadata[key])
    for key in additional_metadata:
      setattr(result_entry_attrs, key, additional_metadata[key])

    # Add the class matrices 
    self.fileh.createArray(result_entry, 'classifications', result.classifications)
    self.fileh.createArray(result_entry, 'goldstandard', result.goldstandard)
    self.fileh.createArray(result_entry, 'instance_indices', result.instance_indices)
     
  ###
  # TokenStream
  ###

  def add_TokenStreams(self, dsname, stream_name, tokenstreams):
    dsnode = getattr(self.datasets, dsname)
    stream_array = self.fileh.createVLArray( dsnode.tokenstreams
                                           , stream_name
                                           , tables.ObjectAtom()
                                           , filters = tables.Filters(complevel=5, complib='zlib') 
                                           )
    for stream in ProgressIter(tokenstreams, label="Adding TokenStreams '%s'" % stream_name):
      stream_array.append(stream)

  def get_TokenStreams(self, dsname, stream_name):
    dsnode = getattr(self.datasets, dsname)
    tsnode = getattr(dsnode.tokenstreams, stream_name)
    return list(t for t in tsnode)

  def list_TokenStreams(self, dsname):
    dsnode = getattr(self.datasets, dsname)
    return set(node._v_name for node in dsnode.tokenstreams)

  ###
  # Sequence
  ###
  def add_Sequence(self, dsname, seq_name, sequence):
    # sequence should arrive as a boolean matrix. axis 0 is parent, axis 1 is child.
    if not issubclass(sequence.dtype.type, numpy.bool_):
      raise ValueError, "sequence must be a boolean matrix"
    if not sequence.shape[0] == sequence.shape[1]:
      raise ValueError, "sequence must be square"

    dsnode = getattr(self.datasets, dsname)
    self._add_sparse_node( dsnode.sequence
                         , seq_name
                         , BoolFeature 
                         , sequence
                         , filters = tables.Filters(complevel=5, complib='zlib') 
                         )

  def get_Sequence(self, dsname, seq_name):
    dsnode = getattr(self.datasets, dsname)
    sqnode = getattr(dsnode.sequence, seq_name)
    # Should be reading each row of the array as a member of a sequence
    # e.g. a row is a thread, each index is the instance index in dataset representing posts
    # returns a list of arrays.
    return self._read_sparse_node(sqnode)

  def list_Sequence(self, dsname):
    dsnode = getattr(self.datasets, dsname)
    return set(node._v_name for node in dsnode.sequence)

  ###
  # Summary
  ###

  def add_Summary(self, tsr_id, interpreter_id, summary, overwrite=False):
    """
    Add a summary, pertaining to a particular interpreter over a 
    particular tsr. Will create the summary node if needed, and
    check for duplicate keys. Overwrite skips the duplicate check
    and immediately updates the keys. 
    """
    tsr_node = getattr(self.results, str(tsr_id))
    if not hasattr(tsr_node, 'summary'):
      self.fileh.createGroup(tsr_node,'summary')
    group_node = tsr_node.summary
    if hasattr(group_node, interpreter_id):
      summary_node = getattr(group_node, interpreter_id)
      if not overwrite:
        # Check for key collisions first 
        old_keys = set(summary_node._v_attrs._v_attrnamesuser)
        new_keys = set(summary.keys())
        overlap = old_keys & new_keys
        if len(overlap) != 0:
          raise ValueError, "Already had the following keys: %s" % str(list(overlap))
    else:
      summary_node = self.fileh.createGroup(group_node, interpreter_id)
    for k,v in summary.iteritems():
      summary_node._v_attrs[k] = v

  def get_Summary(self, tsr_id, interpreter_id):
    """
    Get a summary back from a tsr given an interpreter id
    """
    try:
      group_node = getattr(self.results, str(tsr_id)).summary
      attr_node = getattr(group_node, interpreter_id)._v_attrs
      return dict( (k,attr_node[k]) for k in attr_node._v_attrnamesuser )
    except tables.NoSuchNodeError:
      return {}

  def del_Summary(self, tsr_id, interpreter_id):
    """
    Delete the summary for a given tsr/interpreter combo.
    """
    tsr_node = getattr(self.results, str(tsr_id))
    if not hasattr(tsr_node, 'summary'):
      return False
    group_node = tsr_node.summary
    if hasattr(group_node, interpreter_id):
      summary_node = getattr(group_node, interpreter_id)
      self.fileh.removeNode(summary_node)
      return True
    else:
      return False

  ###
  # Merge
  ###
  def merge(self, other, allow_duplicate=False, do_spaces=True, do_datasets=True, do_tasksets=True, do_results=True):
    """
    Merge the other store's contents into self.
    We can copy tasksets and results verbatim, but spaces and datasets need to 
    take into account a possible re-ordering of features.
    """
    #TODO: May need to organize a staging area to ensure this merge is atomic
    if self.mode == 'r': raise ValueError, "Cannot merge into read-only store"
    ignored_md = ['uuid', 'avg_learn', 'avg_classify', 'name', 'feature_name', 'class_name']

    space_direct_copy = [] # Spaces we copy directly, meaning the featuremap can be copied too
    space_feature_mapping = {}
    if do_spaces or do_datasets:
      # Must do spaces if we do datasets, because spaces may have been updated
      for space_node in ProgressIter(list(other.spaces), label='Copying spaces'):
        logger.debug("Considering space '%s'", space_node._v_name)
        space_name = space_node._v_name
        if hasattr(self.spaces, space_name):
          logger.debug('Already had %s', space_name)
          src_space = other.get_Space(space_name)
          # Need to merge these. Feature spaces can be extended, but there is no mechanism for doing the same with class
          # spaces at the moment, so we must reject any that do not match. 
          dst_space = self.get_Space(space_name)
          if src_space == dst_space:
            logger.debug('  Exact match')
            space_direct_copy.append(space_name)
          else:
            md = get_metadata(space_node)
            if md['type'] == 'class':
              raise ValueError, "Cannot merge due to different versions of %s" % str(md)
            elif md['type'] == 'feature':
              logger.debug('  Attempting to merge %s', str(md))
              # Reconcile the spaces. 
              ## First we need to compute the new features to add
              new_feats = sorted(set(src_space) - set(dst_space))
              logger.debug('    Identified %d new features', len(new_feats))
              reconciled_space = dst_space + new_feats
              if len(new_feats) != 0:
                # Only need to extend if new features are found.
                self.extend_Space(space_name, reconciled_space)
              ## Now we need to build the mapping from the external space to ours
              space_index = dict( (k,v) for v,k in enumerate(reconciled_space))
              space_feature_mapping[space_name] = dict( (i,space_index[s]) for i,s in enumerate(src_space))
            else:
              raise ValueError, "Unknown type of space"
        else:
          self.fileh.copyNode(space_node, newparent=self.spaces)
          space_direct_copy.append(space_name)
        
    if do_datasets:
      for src_ds in ProgressIter(list(other.datasets), label='Copying datasets'):
        dsname = src_ds._v_name

        logger.debug("Considering dataset '%s'", dsname)
        if hasattr(self.datasets, dsname):
          logger.warning("already had dataset '%s'", dsname)
          dst_ds = getattr(self.datasets, dsname)
          # Failure to match instance_id is an immediate reject
          if dst_ds._v_attrs.instance_space != src_ds._v_attrs.instance_space:
            raise ValueError, "Instance identifiers don't match for dataset %s" % dsname
          # The hardest to handle is the feature data, since we may need to rearrange feature maps
        else:
          instance_space = other.get_DatasetMetadata(dsname)['instance_space']
          self.add_Dataset(dsname, instance_space, other.get_Space(dsname))
          dst_ds = getattr(self.datasets, dsname)

        node_names = ['class_data', 'sequence', 'tokenstreams']
        for name in node_names:
          logger.debug('Copying %s',name)
          if hasattr(src_ds, name):
            src_parent = getattr(src_ds, name)
            #TODO: may need to handle incomplete destination nodes
            dst_parent = getattr(dst_ds, name)
            for node in src_parent:
              if hasattr(dst_parent, node._v_name):
                logger.warning("already had '%s' in '%s'", node._v_name, name)
              else:
                self.fileh.copyNode(node, newparent=dst_parent, recursive=True)
          else:
            logger.warning("Source does not have '%s'", name)

        logger.debug('Copying feature_data')
        for node in src_ds.feature_data:
          space_name = node._v_name
          if hasattr(dst_ds.feature_data, space_name):
            logger.warning("already had '%s' in 'feature_data'", space_name) 
          elif space_name in space_direct_copy:
            # Direct copy the feature data because the destination store did not have this
            # space or had exactly this space
            logger.debug("direct copy of '%s' in 'feature_data'", space_name)
            self.fileh.copyNode(node, newparent=dst_ds.feature_data, recursive=True)
          else:
            ax0 = node.feature_map.read(field='ax0')
            ax1 = node.feature_map.read(field='ax1')
            value = node.feature_map.read(field='value')
            feature_mapping = space_feature_mapping[space_name]

            feat_map = [ (i,feature_mapping[j],v) for (i,j,v) in zip(ax0,ax1,value)]
            self.add_FeatureDict(dsname, space_name, feat_map)

      
    # TASKS & RESULTS
    def __merge(datum, check):
      logger.debug("Copying %s", datum)
      for t in ProgressIter(list(getattr(other,datum)), label='Copying %s' % datum):
        logger.debug("Considering %s '%s'", datum, t._v_name)
        md = get_metadata(t)
        for i in ignored_md: 
          if i in md: 
            del md[i]
        if not allow_duplicate and check(md):
          logger.warn("Ignoring duplicate in %s: %s", datum, str(md))
        else:
          try:
            self.fileh.copyNode(t, newparent=getattr(self, datum), recursive=True)
          except tables.NoSuchNodeError:
            logger.critical("Damaged node skipped")

    if do_tasksets:
      __merge('tasksets', self.has_TaskSet)
    if do_results:
      __merge('results', self.has_TaskSetResult)




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

logger = logging.getLogger(__name__)

class StoreError(Exception): pass
class NoData(StoreError): pass
class AlreadyHaveData(StoreError): pass
class InsufficientMetadata(StoreError): pass

def initialize(path, overwrite=False):
  """ Initialize a Store object.
  This is separate from Store.__init__ because the Store constructor
  is used to open an existing store as well.

  ..todo: Refactor this back into Store, deteching if we are starting a new store
  """
  if not overwrite and os.path.exists(path):
    raise IOError, "Refusing to overwrite existing file at %s" % path
  store = Store(path, mode = 'w')
  logger.debug("Creating 'datasets'")
  store.fileh.createGroup( store.root
                        , 'datasets'
                        , 'Per-dataset Data'
                        )

  logger.debug("Creating 'spaces'")
  store.fileh.createGroup( store.root
                        , 'spaces'
                        , 'Space Information'
                        )

  logger.debug("Creating 'tasksets'")
  store.fileh.createGroup( store.root
                        , 'tasksets'
                        , 'TaskSet Data'
                        )

  logger.debug("Creating 'results'")
  store.fileh.createGroup( store.root
                        , 'results'
                        , 'TaskSetResult Data'
                        )

def open_store(path=None, mode='r'):
  """ Convenience function to create a store on demand if needed"""
  try:
    initialize(path, overwrite=False)
  except IOError:
    pass
  return UniversalStore(path, mode=mode)

class Store(object):
  """
  This is the master store class for hydrat. It manages all of the movement of data
  to and from disk.
  """
  def __init__(self, path=None, mode='r', default_path = 'store'):
    self.path = os.path.join(config.get('paths','work'), default_path) if path is None else path
    self.fileh = tables.openFile(self.path, mode=mode)
    self.mode = mode
    logger.debug("Opening Store at '%s', mode '%s'", self.path, mode)
    self.root = self.fileh.root
    super(Store, self).__init__()

  def __del__(self):
    self.fileh.close()

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
    #logger.debug("Reading sparse node")
    values = node.read(field='value')
    feature = node.read(field='feature_index')
    instance = node.read(field='instance_index')
    m = coo_matrix((values,numpy.vstack((instance,feature))), shape=shape)
    #logger.debug('Converting format')
    m = m.tocsr()
    #logger.debug("SPARSE matrix ready")
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
    feature = node.row
    for i,row in enumerate(data):
      for j,v in numpy.vstack((row.indices, row.data)).transpose():
        feature['instance_index'] = i
        feature['feature_index'] = j
        feature['value'] = v
        feature.append()
    self.fileh.flush()
    

class SpaceStore(Store):
  def __init__(self, *args, **kwargs):
    super(SpaceStore, self).__init__(*args, **kwargs)
    self.spaces = self.root.spaces

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
    tags = []
    for space in self.spaces:
      if metadata_matches(space.attrs, desired_metadata):
        tags.append(space._v_name)

    return tags

  #TODO: Use co-operative super stuff to make this awesome
  def get_Metadata(self, tag):
    """
    @param tag: Identifier of the relevant space
    @type tag: uuid
    @rtype: dict of metadata key-value pairs
    """
    tag = str(tag)
    if hasattr(self.spaces, tag):
      space = getattr(self.spaces, tag)
      metadata = dict(   (key, getattr(space._v_attrs,key)) 
                    for  key 
                    in   space._v_attrs._v_attrnamesuser
                    )
      return metadata
    elif hasattr(self.datasets, tag):
      ds = getattr(self.datasets, tag)
      metadata = dict(   (key, getattr(ds._v_attrs,key)) 
                    for  key 
                    in   ds._v_attrs._v_attrnamesuser
                    )
      return metadata
    else:
      raise StoreError, "Unknown tag %s" % tag

  def get_Space(self, tag):
    """
    @param tag: Identifier of the relevant space
    @type tag: uuid
    @rtype: pytables array
    """
    space = getattr(self.spaces, str(tag))
    metadata = self.get_Metadata(tag)
    data = space.read()
    try:
      encoding = metadata['encoding']
    except KeyError:
      logger.warning('Space %s does not have encoding data!', tag)
      encoding = 'nil'
    if encoding != 'nil' and encoding != 'ascii':
      data = [ d.decode(encoding) for d in data ]
    return data

  def list_ClassSpaces(self, dsname = None):
    tags = None
    if dsname is not None:
      ds_tag = self.resolve_Dataset(dsname)
      ds = getattr(self.datasets, ds_tag)
      tags = set(node._v_name for node in ds.class_data)
    s = set()
    for space in self.spaces:
      if tags is None and space._v_attrs.type == 'class':
        s.add(space._v_attrs.name)
      elif tags is not None and space._v_name in tags:
        s.add(space._v_attrs.name)
    return s

  def list_FeatureSpaces(self, dsname = None):
    tags = None
    if dsname is not None:
      ds_tag = self.resolve_Dataset(dsname)
      ds = getattr(self.datasets, ds_tag)
      tags = set(node._v_name for node in ds.feature_data)
    s = set()
    for space in self.spaces:
      if tags is None and space._v_attrs.type == 'feature':
        s.add(space._v_attrs.name)
      elif tags is not None and space._v_name in tags:
        s.add(space._v_attrs.name)
    return s
  
  def add_Space(self, labels, desired_metadata):
    self._check_writeable()
    try:
      self.resolve_Space(desired_metadata)
      raise AlreadyHaveData, "Already have space %s" % desired_metadata
    except NoData:
      pass
    assert 'type' in desired_metadata
    assert 'name' in desired_metadata

    logger.debug( "Adding a %s space '%s' of %d Features"
                    , desired_metadata['type']
                    , desired_metadata['name']
                    , len(labels)
                    )
    id = uuid.uuid4()

    if isinstance(labels[0], unicode):
      encoding = 'utf8'
      labels = [ l.encode(encoding) for l in labels ]
    else:
      encoding = 'ascii'

    desired_metadata['encoding'] = encoding
      
    # We encode all labels because PyTables rejects unicode
    new_space = self.fileh.createArray( self.fileh.root.spaces
                                           , str(id)
                                           , labels
                                           )
    new_space.attrs.date = dt.datetime.now().isoformat() 
    new_space.attrs.uuid = id
    new_space.attrs.size = len(labels)
    for key in desired_metadata:
      setattr(new_space.attrs, key, desired_metadata[key])
    assert metadata_matches(new_space._v_attrs, desired_metadata)
    return id

  def extend_Space(self, labels, space_tag):
    # We do this by checking that the new space is a superset of the
    # old space, with the labels in the same order, then we delete the old 
    # space and add a new space.
    space = self.get_Space(space_tag)
    if any( old != new for old, new in zip(space, labels)):
      raise StoreError, "New labels are not an extension of old labels"

    if len(labels) < len(space):
      raise StoreError, "New labels are less than old labels"

    if len(labels) == len(space):
      logger.debug("Space has not changed, no need to extend")
      return space_tag 

    space_name = self.get_Metadata(space_tag)['name'] 
    logger.debug("Extending '%s' from %d to %d features", space_name, len(space), len(labels))
    metadata = self.get_Metadata(space_tag)

    encoding = metadata['encoding']
    if encoding != 'ascii':
      labels = [l.encode(encoding) for l in labels]

    # Delete the old node
    self.fileh.removeNode( getattr(self.spaces, str(space_tag)) )
    # Create the new node
    new_space = self.fileh.createArray( self.fileh.root.spaces
                                           , str(metadata['uuid'])
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

    return space_tag

class IntFeature(tables.IsDescription):
  instance_index = tables.UInt64Col()
  feature_index  = tables.UInt64Col()
  value          = tables.UInt64Col()

class RealFeature(tables.IsDescription):
  instance_index = tables.UInt64Col()
  feature_index  = tables.UInt64Col()
  value          = tables.Float64Col()

class DatasetStore(SpaceStore):
  def __init__(self, *args, **kwargs):
    super(DatasetStore, self).__init__(*args, **kwargs)
    self.datasets = self.root.datasets

  def list_Datasets(self):
    datasets = set()
    for ds in self.datasets:
      datasets.add(ds._v_attrs.name)
    return datasets

  def resolve_Dataset(self, dsname):
    """
    Resolve a dataset name to its uuid.
    Linear search, assumes dataset names are unique.
    @param dsname: Name of dataset required
    @type dsname: string
    @return dataset's uuid tag 
    """
    for ds in self.datasets:
      if ds._v_attrs.name == dsname:
        return ds._v_name

    raise NoData, 'No dataset %s' % dsname
       
  def instance_identifiers(self, ds_tag):
    """
    @param ds_tag: Tag obtained from resolve_Dataset
    @return: Instance identifiers for dataset
    @rtype: List of strings
    """
    ds = getattr(self.datasets, str(ds_tag))
    return [ i.decode() for i in ds.instance_id ]

  def has_Data(self, ds_tag, space_tag):
    ds = getattr(self.datasets, str(ds_tag))
    return (  hasattr(ds.class_data,   str(space_tag)) 
           or hasattr(ds.feature_data, str(space_tag))
           )

  def get_ClassMap(self, ds_tag, space_tag):
    """
    @param ds_tag: Tag obtained from resolve_Dataset
    @param space_tag: Tag obtained from resolve_Space
    @return: data corresponding to the given dataset in the given class space
    @rtype: pytables array
    """
    ds = getattr(self.datasets, str(ds_tag))
    try:
      class_node = getattr(ds.class_data, str(space_tag)) 
      data = getattr(class_node, 'class_map')
    except AttributeError:
      raise NoData

    metadata = dict()
    metadata['dataset_uuid'] = ds_tag
    metadata['class_uuid']   = space_tag
    return ClassMap(data.read(), metadata)

  def get_FeatureMap(self, ds_tag, space_tag):
    """
    @param ds_tag: Tag obtained from resolve_Dataset
    @param space_tag: Tag obtained from resolve_Space
    @return: data corresponding to the given dataset in the given feature space
    @rtype: varies 
    """
    ds = getattr(self.datasets, str(ds_tag))
    space = self.get_Space(space_tag)
    try:
      feature_node = getattr(ds.feature_data, str(space_tag)) 
    except AttributeError:
      raise NoData

    if feature_node._v_attrs.dense:
      raise StoreError, "Should not be encountering dense FeatureData!"

    data_type = feature_node._v_attrs.type
    logger.debug("Returning SPARSE matrix of type %s", data_type)
    fm = getattr(feature_node, 'feature_map')
    n_inst = len(ds.instance_id) 
    n_feat = len(space)
    m = self._read_sparse_node(fm,shape=(n_inst, n_feat))
    metadata = dict(dataset_uuid=ds_tag, feature_uuid=space_tag)
    return FeatureMap(m, metadata) 

  def get_SizeData(self, ds_tag, space_tag):
    """
    TODO: Generalize this to getInstanceWeightData, which is a vector we can use
    to weight instances in a given feature space. For token-based spaces, this 
    represents the number of tokens in the space.

    @param ds_tag: Tag obtained from resolve_Dataset
    @param space_tag: Tag obtained from resolve_Space
    @return: data corresponding to the size of each instance in this space
    @rtype: pytables array
    """
    ds = getattr(self.datasets, str(ds_tag))
    try:
      feature_node = getattr(ds.feature_data, str(space_tag)) 
      data = getattr(feature_node, 'instance_size')
    except AttributeError:
      raise NoData
    return data.read()

  def get_Data(self, dsname, space_metadata):
    """
    Convenience method which combines the tag resolution step
    with the data access step.
    """
    ds_tag = self.resolve_Dataset(dsname)

    space_tag = self.resolve_Space(space_metadata)
    s_type = space_metadata['type']
    s_name = space_metadata['name']

    if s_type == 'class':
      data = self.get_ClassMap(ds_tag, space_tag)
      ds_meta = self.get_Metadata(ds_tag)
      cl_meta = self.get_Metadata(space_tag)
      data.metadata['dataset']      = ds_meta['name']
      data.metadata['class_name']   = cl_meta['name']
    elif s_type == 'feature':
      data = self.get_FeatureMap(ds_tag, space_tag)
      data.metadata['feature_desc']+= (s_name,)
    else:
      raise StoreError, "Unknown data type: %s" % s_type

    logger.debug("Retrieved %s data for '%s' in space '%s'", s_type, dsname, s_name)
    return data 

  def add_Dataset(self, instance_ids, name):
    self._check_writeable()
    try:
      self.resolve_Dataset(name)
      raise AlreadyHaveData, "Already have dataset by name %s", name
    except NoData:
      pass

    id = uuid.uuid4()

    # Create a group for the DataSet
    ds = self.fileh.createGroup( self.fileh.root.datasets
                               , str(id)
                               )

    attrs = ds._v_attrs

    # Note down our metadata
    attrs.name              = name
    attrs.date              = dt.datetime.now().isoformat()
    attrs.uuid              = id
    attrs.num_instances     = len(instance_ids)

    # Create the instance_id array
    self.fileh.createArray( ds 
                          , "instance_id"
                          , numpy.array([i.encode() for i in instance_ids])
                          , title = "Instance Identifiers"
                          )

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
    return id

  def add_FeatureDict(self, ds_tag, space_tag, feat_map):
    self._check_writeable()
    ds_name = self.get_Metadata(ds_tag)['name']
    space_name = self.get_Metadata(space_tag)['name']
    logger.debug("Adding feature map to dataset '%s' in space '%s'", ds_name, space_name)
    ds = getattr(self.datasets, str(ds_tag))
    space = getattr(self.spaces, str(space_tag))

    group =  self.fileh.createGroup( ds.feature_data
                                        , space._v_name
                                        , "Sparse Feature Map %s" % space._v_name
                                        )
    group._v_attrs.date  = dt.datetime.now().isoformat()
    group._v_attrs.dense = False 
    group._v_attrs.type  = 'int' if all(isinstance(i[2], int) for i in feat_map) else 'float'

    fm_node = self.fileh.createTable( group
                                    , 'feature_map'
                                    , IntFeature if group._v_attrs.type == 'int' else RealFeature 
                                    , title = 'Sparse Feature Map'
                                    , filters = tables.Filters(complevel=5, complib='zlib') 
                                    , expectedrows = len(feat_map)
                                    )

    # Initialize space to store instance sizes.
    n_inst = len(self.instance_identifiers(ds_tag))
    instance_sizes = numpy.zeros(n_inst, dtype='uint64')
    
    attrs = fm_node._v_attrs
    setattr(attrs, 'dtype', group._v_attrs.type)
    setattr(attrs, 'shape', (n_inst, len(space)))

    # Add the features to the table
    feature = fm_node.row
    for i, j, v in feat_map:
      feature['instance_index'] = i
      feature['feature_index'] = j
      feature['value'] = v
      feature.append()
      instance_sizes[i] += v # Keep tally of instance size

    # Store the instance sizes
    fm_sizes = self.fileh.createArray( group
                                     , 'instance_size'
                                     , instance_sizes
                                     , title = 'Instance Sizes'
                                     )

    self.fileh.flush()

  def add_FeatureMap(self, ds_tag, space_tag, feat_map):
    self._check_writeable()
    ds_name = self.get_Metadata(ds_tag)['name']
    space_name = self.get_Metadata(space_tag)['name']
    logger.debug("Adding feature map to dataset '%s' in space '%s'", ds_name, space_name)
    ds = getattr(self.datasets, str(ds_tag))
    space = getattr(self.spaces, str(space_tag))

    group =  self.fileh.createGroup( ds.feature_data
                                        , space._v_name
                                        , "Sparse Feature Map %s" % space._v_name
                                        )
    group._v_attrs.date  = dt.datetime.now().isoformat()
    group._v_attrs.dense = False 
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


  def add_ClassMap(self, ds_tag, space_tag, class_map):
    self._check_writeable()
    ds_name = self.get_Metadata(ds_tag)['name']
    space_name = self.get_Metadata(space_tag)['name']
    logger.debug("Adding Class Map to dataset '%s' in space '%s'", ds_name, space_name)
    ds = getattr(self.datasets, str(ds_tag))
    space = getattr(self.spaces, str(space_tag))

    num_docs = len(ds.instance_id)
    num_classes = len(space)

    # Check that the map is of the right shape for the dataset and space
    if class_map.shape != (num_docs, num_classes):
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
                               

class TaskStore(Store):
  def __init__(self, *args, **kwargs):
    super(TaskStore, self).__init__(*args, **kwargs)
    self.tasksets = self.root.tasksets

  def add_TaskSet(self, taskset, additional_metadata={}):
    self._check_writeable()
    try:
      taskset_uuid = taskset.metadata['uuid']
    except KeyError:
      taskset_uuid = uuid.uuid4()
      taskset.metadata['uuid'] = taskset_uuid

    if 'date' not in taskset.metadata:
      taskset.metadata['date'] = dt.datetime.now().isoformat()

    taskset_entry_tag = str(taskset_uuid)
    taskset_entry = self.fileh.createGroup(self.tasksets, taskset_entry_tag)
    taskset_entry_attrs = taskset_entry._v_attrs

    for key in taskset.metadata:
      setattr(taskset_entry_attrs, key, taskset.metadata[key])
    for key in additional_metadata:
      setattr(taskset_entry_attrs, key, additional_metadata[key])

    logger.debug('Adding a taskset %s %s', str(taskset.metadata), str(additional_metadata))

    for i,task in enumerate(ProgressIter(taskset.tasks, label="Adding Tasks")):
      self._add_Task(task, taskset_entry, dict(index=i))
    self.fileh.flush()

    return taskset_entry_tag

  def new_TaskSet(self, taskset):
    """Convenience method which checks no previous taskset has matching 
    metadata. Useful for situations where we build up the tasksets incrementally
    """
    try:
      self.get_TaskSet(taskset.metadata)
      raise AlreadyHaveData, "Already have taskset %s" % str(taskset.metadata)
    except NoData:
      pass
    self.add_TaskSet(taskset)
      

  def _add_Task(self, task, ts_entry, additional_metadata={}): 
    self._check_writeable()
    try:
      task_uuid = task.metadata['uuid']
    except KeyError:
      task_uuid = uuid.uuid4()
      task.metadata['uuid'] = task_uuid
    task_tag = str(task_uuid)

    # Create a group for the task
    task_entry = self.fileh.createGroup(ts_entry, task_tag)
    task_entry_attrs = task_entry._v_attrs

    # Add the metadata
    for key in task.metadata:
      setattr(task_entry_attrs, key, task.metadata[key])
    for key in additional_metadata:
      setattr(task_entry_attrs, key, additional_metadata[key])

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
                         , IntFeature if issubclass(tr.dtype.type,numpy.int) else RealFeature 
                         , tr
                         , filters = tables.Filters(complevel=5, complib='zlib') 
                         )
    self._add_sparse_node( task_entry
                         , 'test_vectors'
                         , IntFeature if issubclass(te.dtype.type,numpy.int) else RealFeature 
                         , te 
                         , filters = tables.Filters(complevel=5, complib='zlib') 
                         )
    self.fileh.flush()

  def get_TaskSet(self, desired_metadata):
    """ Convenience function to bypass tag resolution """
    tags = self._resolve_TaskSet(desired_metadata)
    if len(tags) == 0: raise NoData
    elif len(tags) > 1: raise InsufficientMetadata
    try:
      return self._get_TaskSet(tags[0])
    except tables.NoSuchNodeError:
      logger.warning('Removing damaged TaskSet node with metadata %s', str(desired_metadata))
      self.fileh.removeNode(self.tasksets, tags[0], recursive=True)
      raise NoData


  def _get_TaskSet(self, taskset_tag):
    try:
      taskset_entry  = getattr(self.tasksets, taskset_tag)
    except AttributeError:
      raise KeyError, str(taskset_tag)
    metadata = get_metadata(taskset_entry)
    tasks = []
    for task_entry in taskset_entry._v_groups.values():
      tasks.append(self._get_Task(task_entry))
    tasks.sort(key=lambda r:r.metadata['index'])
    return TaskSet(tasks, metadata)

  def _get_Task(self,task_entry): 
    metadata = get_metadata(task_entry)
    t = Task()
    t.metadata = metadata
    t.train_classes = task_entry.train_classes.read()
    t.train_indices = task_entry.train_indices.read()
    t.test_classes  = task_entry.test_classes.read()
    t.test_indices  = task_entry.test_indices.read()
    t.train_vectors = self._read_sparse_node(task_entry.train_vectors)
    t.test_vectors  = self._read_sparse_node(task_entry.test_vectors)
    return t

  def _resolve_TaskSet(self, desired_metadata):
    """Returns all tags whose entries match the supplied metadata"""
    desired_keys = []
    for node in self.tasksets._v_groups:
      attrs = getattr(self.tasksets,node)._v_attrs
      if metadata_matches(attrs, desired_metadata):
        desired_keys.append(node)
    return desired_keys

class ResultStore(SpaceStore):
  def __init__(self, *args, **kwargs):
    super(ResultStore, self).__init__(*args, **kwargs)
    self.results = self.root.results

  def get_TaskSetResult(self, desired_metadata):
    """ Convenience function to bypass tag resolution """
    tags = self._resolve_TaskSetResults(desired_metadata)
    if len(tags) == 0: raise NoData
    elif len(tags) > 1: raise InsufficientMetadata
    return self._get_TaskSetResult(tags[0])

  def _get_TaskSetResult(self, tsr_tag):
    try:
      tsr_entry  = getattr(self.results, tsr_tag)
    except AttributeError:
      raise KeyError, str(tsr_tag)
    metadata = get_metadata(tsr_entry)
    results = []
    for result_entry in tsr_entry._v_groups.values():
      results.append(self._get_Result(result_entry))
    results.sort(key=lambda r:r.metadata['index'])
    return TaskSetResult(results, metadata)

  def _get_Result(self, result_entry):
    metadata = get_metadata(result_entry)
    goldstandard     = result_entry.goldstandard.read()
    classifications  = result_entry.classifications.read()
    instance_indices = result_entry.instance_indices.read()
    return Result( goldstandard
                 , classifications
                 , instance_indices
                 , metadata
                 )

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


class UniversalStore(DatasetStore, TaskStore, ResultStore):
  def __init__(self, *args, **kwargs):
    super(UniversalStore, self).__init__(*args, **kwargs)


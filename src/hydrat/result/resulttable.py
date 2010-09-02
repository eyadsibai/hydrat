import tables
import uuid
import datetime
import logging
import uuid
import datetime as dt

from hydrat.result.result import Result
from hydrat.result.tasksetresult import TaskSetResult
from hydrat.common.metadata import metadata_matches, get_metadata


class ResultTableReader(object):
  def __init__(self, path):
    self.logger = logging.getLogger('hydrat.result.resulttable.ResultTableReader')
    self.table_file = tables.openFile(path, mode = 'r')

  def __del__(self):
    self.close()

  def close(self):
    self.table_file.close()

  def __len__(self):
    return len(self.table_file.root._v_groups)

  def __contains__(self, item):
    return item in self.table_file.root._v_groups
    
  def __getitem__(self, key):
    root       = self.table_file.root
    try:
      tsr_entry  = getattr(root, key)
    except AttributeError:
      raise KeyError, str(key)
    return self._get_TaskSetResult(tsr_entry) 

  def get_TaskSetResult(self, tsr_tag):
    raise DeprecatedWarning, "Use the subscript interface"

  def _get_TaskSetResult(self, tsr_entry):
    metadata = get_metadata(tsr_entry)
    results = []
    for result_entry in tsr_entry._v_groups.values():
      results.append(self._get_Result(result_entry))

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

  def get_tags(self):
    root = self.table_file.root
    keys = [ k for k in root._v_groups.keys() if k.startswith('tsr')]
    return keys

  def get_metadata_map(self):
    map = {}
    for tag in self.get_tags():
      tsr = self[tag]
      for key in tsr.metadata:
        value = tsr.metadata[key]
        try:
          map.setdefault(key,{}).setdefault(value,set()).add(tag)
        except TypeError:
          pass
    return map
    

  def resolve_tag(self, desired_metadata):
    """Returns all tags whose entries match the supplied metadata"""
    root       = self.table_file.root

    return [    tag 
           for  tag 
           in self.get_tags() 
           if metadata_matches( getattr(root, tag)._v_attrs, desired_metadata) 
           ]
  
  def get_Space(self, id):
    spaces = self.table_file.root.spaces
    for space in spaces:
      if space._v_name == str(id):
        data = []
        for d in space.read():
          try:
            data.append(d.decode())
          except AttributeError:
            data.append(unicode(d))
        return data 
    raise ValueError, "No space with id %s" % id

  # adapted from preprocessor.model.inducer.token_inducer
  def resolve_spaces(self, desired_metadata):
    spaces = self.table_file.root.spaces
    candidate_spaces = [    space 
                       for  space 
                       in   spaces
                       if   metadata_matches(space._v_attrs, desired_metadata)
                       ]

    if len(candidate_spaces) == 0:
      self.logger.info("No space matches this description yet")
      return []
    self.logger.info("Found %d candidate spaces", len(candidate_spaces))
    return candidate_spaces
     

class ResultTableWriter(ResultTableReader):
  def __init__(self, path):
    self.logger = logging.getLogger('hydrat.result.resulttable.ResultTableWriter')
    self.table_file = tables.openFile(path, mode = 'a')
    root = self.table_file.root
    if not hasattr(root, 'spaces'):
      self.table_file.createGroup(root, 'spaces')
      

  def __del__(self):
    self.close()

  # Mechanism for "trashing" a tag.
  # This is a just a last defence against accidental destruction of data.
  # There is no mechanism for untrashing at the moment.
  def del_TaskSetResult(self, tag):
    root = self.table_file.root
    if "trash" not in root:
      self.logger.info('Creating Trash group')
      self.table_file.createGroup(root, "trash", title="Trashed TSRs")
    self.logger.info('Trashing %s' % tag)
    self.table_file.moveNode(root, root.trash, name=tag)

  def add_TaskSetResult(self, tsr, additional_metadata={}):
    try:
      tsr_uuid = tsr.metadata['uuid']
    except KeyError:
      tsr_uuid = uuid.uuid4()
      tsr.metadata['uuid'] = tsr_uuid

    if 'date' not in tsr.metadata:
      tsr.metadata['date'] = datetime.datetime.now().isoformat()

    tsr_entry_tag = 'tsr_'+str(tsr_uuid).replace('-','')
    root = self.table_file.root
    tsr_entry = self.table_file.createGroup(root, tsr_entry_tag)
    tsr_entry_attrs = tsr_entry._v_attrs

    for key in tsr.metadata:
      setattr(tsr_entry_attrs, key, tsr.metadata[key])
    for key in additional_metadata:
      setattr(tsr_entry_attrs, key, additional_metadata[key])

    for result in tsr.raw_results:
      self.add_Result(result, tsr_entry)
    self.table_file.flush()
    return tsr_entry_tag

  def add_Result(self, result, tsr_entry):
    try:
      result_uuid = result.metadata['uuid']
    except KeyError:
      result_uuid = uuid.uuid4()
      result.metadata['uuid'] = result_uuid
    result_tag = 'result_'+str(result_uuid).replace('-','')

    # Create a group for the result
    result_entry = self.table_file.createGroup(tsr_entry, result_tag)
    result_entry_attrs = result_entry._v_attrs

    # Add the metadata
    for key in result.metadata:
      setattr(result_entry_attrs, key, result.metadata[key])

    # Add the class matrices 
    self.table_file.createArray(result_entry, 'classifications', result.classifications)
    self.table_file.createArray(result_entry, 'goldstandard', result.goldstandard)
    self.table_file.createArray(result_entry, 'instance_indices', result.instance_indices)

  def has_Space(self, id):
    return hasattr(self.table_file.root.spaces, str(id))
    
  #adapted from preprocessor/model/inducer/token_inducer.py
  def add_Space(self, data, metadata, id = None):
    """
    Some experiments require that additional spaces be created, for example
    if only a subset of classes is used. We may not always want to store
    these spaces with the model itself, so instead we provide a facility
    in the resulttable to store them.
    """
    # Generate an ID for the space if one was not provided.
    id = uuid.uuid4() if id is None else id
    self.logger.info("Adding space with id %s", id)
    if self.has_Space(id):
      # Already have this space
      self.logger.info("Already had space %s!", id)
      # TODO: Some kind of metadata consistency check
      return id
    new_space = self.table_file.createArray( self.table_file.root.spaces
                                           , str(id)
                                           , [unicode(d).encode() for d in data]
                                           )
    new_space.attrs.date = dt.datetime.now().isoformat() 
    new_space.attrs.uuid = id
    for key in metadata:
      setattr(new_space.attrs, key, metadata[key])
    return id

def merge(src, dst):
  """Merges one result table into another"""
  logger = logging.getLogger('hydrat.result.resulttable.merge')
  assert isinstance(src, ResultTableReader)
  assert isinstance(dst, ResultTableWriter)

  src_root = src.table_file.root
  dst_root = dst.table_file.root

  skipcount = 0
  copycount = 0

  logger.info( "Merging %s into %s"
             , src.table_file.filename
             , dst.table_file.filename
             )

  for node in src_root:
    tag = node._v_name
    if tag.startswith('tsr'):
      if hasattr(dst_root, tag):
        logger.debug("Destination already has %s, skipping", tag)
        skipcount += 1
      else:
        logger.debug("Copying %s", tag)
        node._f_copy( newparent = dst_root
                    , recursive = True
                    )
        copycount += 1

  logger.info("Merging spaces")
  for space in src_root.spaces:
    tag = space._v_name
    if hasattr(dst_root.spaces, tag):
      #TODO: Sanity check upon space collision
      logger.debug("Destination already has space %s, skipping", tag)
    else:
      logger.debug("Copying %s", tag)
      space._f_copy( newparent = dst_root.spaces
                   , recursive = True
                   )

  logger.info("%d results copied, %d skipped", copycount, skipcount)

      




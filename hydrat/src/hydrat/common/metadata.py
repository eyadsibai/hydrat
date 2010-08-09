# Common operations on metadata stored in pytables nodes

def metadata_matches(attrs, desired_metadata):
  for key in desired_metadata:
    try:
      value = getattr(attrs, key)
    except AttributeError:
      # No match if node does not have this attribute
      return False
    if value != desired_metadata[key]:
      # No match if node's value does not match the desired value
      return False
  # Match if we got this far
  return True

def get_metadata(node):
  metadata   = dict(    ( key, getattr(node._v_attrs, key) ) 
                    for  key
                    in   node._v_attrs._f_list()
                    )
  return metadata

import numpy
def map2matrix(mapping, instance_ids=None, labels=None):
  #TODO: Sanity checks on instance_ids and labels
  if instance_ids is None:
    instance_ids = sorted(mapping) # Use the sorted keys, since mappings are unordered
  if labels is None:
    labels = reduce(set.union, (set(d) for d in mapping.itervalues()))
  matrix = numpy.zeros( (len(instance_ids), len(labels)),dtype=bool)
  indices = dict( (k,v) for v, k in enumerate(labels))

  for i, id in enumerate(instance_ids):
    for c in mapping[id]:
      j = indices[c]
      matrix[i,j] = True

  return matrix

def matrix2map(matrix, instance_ids, labels):
  assert len(instance_ids), len(labels) == matrix.shape
  labels = numpy.array(labels)
  mapping = {}
  for i, id in enumerate(instance_ids):
    row = matrix[i].nonzero()[0]
    id_labels = list(labels[row])
    mapping[id] = id_labels
  return mapping

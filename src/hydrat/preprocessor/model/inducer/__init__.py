import numpy
from collections import defaultdict

def invert_text(text, tokenizer):
  fv = defaultdict(int)
  for token in tokenizer(text):
    fv[token] += 1
  return fv

def invert_map(instance_map, tokenizer):
  fvs = [ invert_text(instance_map[id], tokenizer) for id in instance_map ]
  return fvs
      
def dense_repr(sparse_data, features):
  matrix = numpy.zeros((len(sparse_data), len(features)),  dtype='int64')
  feature_indices = dict( (k, v) for v, k in enumerate(features))
  for i, fv in enumerate(sparse_data):
    for f in fv:
      j = feature_indices[f]
      matrix[i,j] = fv[f]
  return matrix

import numpy
def map2matrix(mapping, instance_ids=None, labels=None):
  #TODO: Sanity checks on instance_ids and labels
  if instance_ids is None:
    instance_ids = sorted(mapping) # Use the sorted keys, since mappings are unordered
  if labels is None:
    labels = reduce(set.union, (set(d) for d in mapping.itervalues()))
  matrix = numpy.zeros( (len(mapping), len(labels)),dtype=bool)
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
    

# Has been renamed to map2matrix
from hydrat.common.decorators import deprecated
@deprecated
def class_matrix(class_map, instance_ids, classlabels):
  return map2matrix(class_map, instance_ids, classlabels)

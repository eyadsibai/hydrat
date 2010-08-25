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

def class_matrix(class_map, instance_ids, classlabels):
  matrix = numpy.zeros( (len(class_map), len(classlabels)),dtype=bool)
  class_indices = dict( (k,v) for v, k in enumerate(classlabels))

  for i, id in enumerate(instance_ids):
    for c in class_map[id]:
      j = class_indices[c]
      matrix[i,j] = True

  return matrix


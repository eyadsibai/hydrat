import hydrat
import scipy.sparse

"""
Convert sequencing information from a compact list-of-lists-of-indices representation
to a sparse matrix representation suitable for computation (and back).
Single-item sequences should not be present
"""
def sequence2follow(sequence, size=None):
  if size is None:
    # Try to infer size of collection from the largest identifier in sequence
    size = max( max(s) for s in sequence) + 1
  matrix = scipy.sparse.dok_matrix((size, size), dtype='bool')
  for subseq in sequence:
    if len(subseq) == 1: raise ValueError, "Do not support single-item sequences"
    for i in xrange(len(subseq)-1):
      parent, child = subseq[i:i+2]
      matrix[child, parent] = True
    
  return scipy.sparse.csr_matrix(matrix, dtype=matrix.dtype)
  
def follow2sequence(matrix):
  matrix = scipy.sparse.dok_matrix(matrix, dtype=matrix.dtype)
  pairs = sorted((p,c) for (c,p),b in matrix.items())
  seq = []
  i = iter(pairs)
  subseq = list(i.next())
  for parent, child in i:
    if subseq[-1] == parent:
      subseq.append(child)
    else:
      seq.append(scipy.array(subseq, dtype='uint64'))
      subseq = [parent, child]
  seq.append(scipy.array(subseq, dtype='uint64'))
  return seq

if __name__ == "__main__":
  import cPickle
  seq = cPickle.load(open('sequence'))
  mat = sequence2follow(seq)
  seq2 = follow2sequence(mat)
  mat2 = sequence2follow(seq2)
  assert mat.todok().items() == mat2.todok().items()

# Feature extractors
# Accepts a tokenstream, produces a feature map

from hydrat.common.tokenizers import NGram
from hydrat.common.counter import Counter
def ngram_dist(n, ts):
  if isinstance(ts, list):
    # Need to handle lists specially as a slice of a list is a list,
    # which is unhashable and thus incompatible with counter.
    return Counter(NGram(1)(tuple(ts)))
  else:
    return Counter(NGram(1)(ts))

def unigram(ts):   return ngram_dist(1, ts) 
def bigram(ts):    return ngram_dist(2, ts) 
def trigram(ts):   return ngram_dist(3, ts)
def quadgram(ts):  return ngram_dist(4, ts)
def pentagram(ts): return ngram_dist(5, ts)

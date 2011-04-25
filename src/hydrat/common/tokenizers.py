"""tokenizer module contains tokenizer functions that take a string and return
a token generator or a list of tokens."""

def whiteSpace(string):
  return string.split()

def bag_of_words(string):
  from stringtools import strip_punctuation
  return strip_punctuation(string.lower()).split()

def stemmed(string):
  from PorterStemmer import PorterStemmer
  p = PorterStemmer()
  def stemword(word): return p.stem(word, 0, len(word)-1)
  tokens = []
  for word in basic(string):
    tokens.append(stemword(word))
  return tokens

def dotw(string):
  from stringtools import strip_html
  return numberToken(basic(strip_html(string)))

def numberToken(tokenlist):
  return map(lambda x: "NUMBER" if x.isdigit() else x, tokenlist)

def perword_bitrigram(string):
  return perword_bigram(string) + perword_trigram(string)

def perword_bigram(string):
  return perword_ngram(2, string)

def perword_trigram(string):
  return perword_ngram(3, string)

def perword_ngram(n, string):
  from stringtools import strip_punctuation
  tokens = []
  for word in string.split():
    word = filter(unicode.isalpha,word)
    word = "_" + word.lower() + "_"
    for pos in xrange(len(word)-n+1):
      tokens.append(word[pos:pos+n])
  return tokens

def word_ngram(n, string):
  tokens = []
  for ngram in stream_ngram(n, string.split()):
    tokens.append( "_".join(ngram))
  return tokens

def word_bigram(str): return word_ngram(2,str)
def word_trigram(str): return word_ngram(3,str)

def stream_ngram(n, list):
  result = []
  for pos in xrange(len(list) - n + 1):
    result.append(list[pos:pos+n])
  return result

def stream_unigram(str): return stream_ngram(1,str)
def stream_bigram(str): return stream_ngram(2,str)
def stream_trigram(str): return stream_ngram(3,str)
def stream_unibigram(str):  return stream_unigram(str) + stream_bigram(str)
def stream_bitrigram(str):  return stream_bigram(str) + stream_trigram(str)

all_stream = [ stream_unigram
             , stream_bigram
             , stream_trigram
             ]

def bytes(str):
  return filter(lambda x: x != '\x00', str.encode('unicode_internal'))

def byte_unigram(str):    return stream_unigram(bytes(str))
def byte_bigram(str):     return stream_bigram(bytes(str))
def byte_trigram(str):    return stream_trigram(bytes(str))
def byte_unibigram(str):  return byte_unigram(str) + byte_bigram(str)
def byte_bitrigram(str):  return byte_bigram(str) + byte_trigram(str)
    
all_byte = [ byte_unigram
           , byte_bigram
           , byte_trigram
           ]

def codepoints(string, encoding = 'utf-8'):
  return string.decode(encoding, 'ignore')

def codepoint_unigram(str):    return stream_unigram(codepoints(str))
def codepoint_bigram(str):     return stream_bigram(codepoints(str))
def codepoint_trigram(str):    return stream_trigram(codepoints(str))
def codepoint_unibigram(str):  return codepoint_unigram(str) + codepoint_bigram(str)
def codepoint_bitrigram(str):  return codepoint_bigram(str) + codepoint_trigram(str)
    
all_codepoint = [ codepoint_unigram
                , codepoint_bigram
                , codepoint_trigram
                ]

from collections import defaultdict, deque, Callable
class Tokenizer(Callable): pass

import itertools
def ngram(n, seq):
  tee = itertools.tee(seq, n)
  for i in xrange(n):
    for j in xrange(i):
      # advance iterators, ignoring result
      tee[i].next()
  while True:
    token = tuple(t.next() for t in tee)
    if len(token) < n: break
    yield token


class NGram(Tokenizer):
  def __init__(self, n):
    self.n = n
    self.__name__ = '%d-gram' % n

  def __call__(self, seq):
    try:
      return [ seq[pos:pos+self.n] for pos in xrange(len(seq) - self.n + 1) ]
    except TypeError:
      # Handle generators
      return ngram(self.n, seq)

class Scanner(Tokenizer):
  """
  Implementation of Aho-Corasick string matching.
  This class should be instantiated with a set of keywords, which
  will then be the only tokens generated by the class's search method,
  """
  def __init__(self, keywords, alphabet=None):
    self.built = False
    self.keywords = keywords 
    if alphabet is None:
      self.alphabet = map(chr, range(1<<8))
    self.build(keywords)

  def __call__(self, value):
    return self.search(value)

  def build(self, keywords):
    goto = dict()
    fail = dict()
    output = defaultdict(set)

    # Algorithm 2
    newstate = 0
    for a in keywords:
      state = 0
      j = 0
      while (state, a[j]) in goto:
        state = goto[(state, a[j])]
        j += 1
      for p in range(j, len(a)):
        newstate += 1
        goto[(state, a[p])] = newstate
        #print "(%d, %s) -> %d" % (state, a[p], newstate)
        state = newstate
      output[state].add(a)
    for a in self.alphabet:
      if (0,a) not in goto: 
        goto[(0,a)] = 0

    # Algorithm 3
    queue = deque()
    for a in self.alphabet:
      if goto[(0,a)] != 0:
        s = goto[(0,a)]
        queue.append(s)
        fail[s] = 0
    while queue:
      r = queue.popleft()
      for a in self.alphabet:
        if (r,a) in goto:
          s = goto[(r,a)]
          queue.append(s)
          state = fail[r]
          while (state,a) not in goto:
            state = fail[state]
          fail[s] = goto[(state,a)]
          #print "f(%d) -> %d" % (s, goto[(state,a)]), output[fail[s]]
          if output[fail[s]]:
            output[s].update(output[fail[s]])

    # Algorithm 4
    self.nextmove = {}
    for a in self.alphabet:
      self.nextmove[(0,a)] = goto[(0,a)]
      if goto[(0,a)] != 0:
        queue.append(goto[(0,a)])
    while queue:
      r = queue.popleft()
      for a in self.alphabet:
        if (r,a) in goto:
          s = goto[(r,a)]
          queue.append(s)
          self.nextmove[(r,a)] = s
        else:
          self.nextmove[(r,a)] = self.nextmove[(fail[r],a)]

    # convert the output to tuples, as tuple iteration is faster
    # than set iteration
    self.output = dict((k, tuple(output[k])) for k in output)

  def search(self, string):
    state = 0
    for letter in string:
      state = self.nextmove[(state, letter)]
      for key in self.output.get(state, []):
        yield key


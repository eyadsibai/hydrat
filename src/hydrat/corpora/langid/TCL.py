from __future__ import with_statement
import os
import chardet
from hydrat.preprocessor.dataset.text import ByteUBTQP
from hydrat.preprocessor.dataset.encoded import CodepointUBT
from hydrat import configuration

from iso639 import ISO639_1
import language_data.iso639 as langdata

def compute_encodings(tclpath, efile ='encodings'):
  encpath = os.path.join(tclpath, efile )
  GOLDpath = os.path.join(tclpath, 'GOLD')
  _encodings = {}
  with open(GOLDpath) as f:
    for line in f:
      if '/' in line:
        info, filename = line.strip().split('/',1)
        encoding = (info.split('_',1)[1]).lower()
        _encodings.setdefault(filename,[]).append(encoding)
  with open(encpath, 'w') as enc:
    for filename in _encodings:
      e = _encodings[filename]
      oe = e[:]
      if len(e) == 1:
        unique_e = e[0]
      else:
        with open(os.path.join(tclpath, filename)) as f:
          detected_encoding = chardet.detect(f.read())['encoding']
        if detected_encoding == 'utf-8':
          unique_e = 'utf-8'
        else:
          try:
            e.remove('utf-8')
            e.remove('unicode')
          except ValueError:
            pass
          if len(e) == 1:
            unique_e = e[0]
          else:
            unique_e = detected_encoding 
      #if unique_e == None: print "########## None!!"
      #print "Disamb '%s'"%filename,"between",oe,"det",detected_encoding,"sel",unique_e
      print >>enc, "%s %s" % (filename, unique_e)

  
class TCL(ByteUBTQP, CodepointUBT, ISO639_1):
  __name__ = "TCL"
  tclpath = configuration.get('corpora',"TCL")

  def encodings(self):
    _encodings = {}
    encodingpath = os.path.join(self.tclpath, "encodings")

    if not os.path.exists(encodingpath):
      compute_encodings(self.tclpath, "encodings")

    with open(encodingpath) as f:
      for line in f:
        instance_id, e = line.split(' ',1)
        _encodings[instance_id] = e.strip()
    return _encodings

  def text(self):
    instances = {}
    classpath = os.path.join(self.tclpath, "GOLD.2")
    with open(classpath) as f:
      for line in f:
        if '|' in line:
          instance_id = line.strip().split('|',1)[1].rsplit('/',1)[1]
          path = os.path.join(self.tclpath, instance_id)
          # Reject files smaller than 10 bytes
          if os.path.getsize(path) > 10:
            with open(path) as file:
              instances[unicode(instance_id)] = file.read()
    return instances

  def cm_tcl_lang(self):
    cm = {}
    classpath = os.path.join(self.tclpath, "GOLD.2")
    with open(classpath) as f:
      for line in f:
        if '|' in line:
          language, instance_info = line.strip().split('|',1)
          instance_id = instance_info.rsplit('/',1)[1]
          path = os.path.join(self.tclpath, instance_id)
          # Reject files smaller than 10 bytes
          if os.path.getsize(path) > 10:
            cm[unicode(instance_id)] = [language]
    return cm 

  def cm_iso639_1(self):
    additional_mappings =\
      { 'Greek' : 'el'
      , 'Swahili' : 'sw'
      , 'Slovene' : 'sl'
      , 'Azeri' : 'az'
      , 'Malay' : 'ml'
      , 'Pashto' : 'ps'
      , 'Greatlakes' : 'UNKNOWN'
      , 'Kyrgyz' : 'ky'
      , 'Laothian' : 'lo'
      }
    tcl_639_mapping = dict( (k,v) for v,k in langdata.iso639_1_name_assoc)
    tcl_639_mapping.update(additional_mappings)
    cm = self.cm_tcl_lang()
    for id in cm:
      cm[id] = [ tcl_639_mapping[label] for label in cm[id]]
    return cm

if __name__ == "__main__":
  from hydrat.common.log import getHydraLogger
  l = getHydraLogger()
  #x = TCL()
  #print x.encodings()
  compute_encodings(TCL.tclpath)

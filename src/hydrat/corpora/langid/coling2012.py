import os
from hydrat import config
from hydrat.dataset.text import FilePerClass
from hydrat.dataset.iso639 import ISO639_1
from hydrat.configuration import Configurable, DIR

filename_map = {
  'dnevniavaz.ba.200.check' : 'bs',
  'politika.rs.200.check'   : 'sr',
  'vecernji.hr.200.check'   : 'hr',
  }

  

class BHSTiedemann(Configurable, ISO639_1, FilePerClass):
  """
  hydrat corpora interface to Jorg Tiedemann and Nikola Ljubesic's 
  bs/hr/sr evaluation data from their COLING 2012 paper.
  """
  requires=\
    { ('corpora', 'tiedemann-coling2012-bhs') : DIR('tiedemann-bs.hr.sr')
    }

  def data_path(self):
    return os.path.join(config.getpath('corpora','tiedemann-coling2012-bhs'), 'eval')

  def cm_iso639_1(self):
    return self.filename2class(filename_map)


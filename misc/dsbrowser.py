import cherrypy
import sys
import re
import markup
from hydrat.common.invert_dict import invert_dict
newlines = re.compile('\n')

def pretty(raw_text):
  return newlines.sub('<br/>', str(raw_text))

def html_list(seq):
  out = ""
  out.append('<ul>')
  for item in seq:
    out.append('<li>%s</li>'%str(item))
  out.append('</ul>')
  return out
  
class TextDatasetBrowser(object):
  def __init__(self, ds):
    self.ds = ds
    self._text = ds.text()
    self.class_index = dict((c,invert_dict(ds.classmap(c)))  for c in ds.classmap_names)

  @cherrypy.expose
  def index(self):
      return str(self.ds)

  @cherrypy.expose
  def instances(self):
    ids = self.ds.instance_ids
    links = [self.instance_link(i) for i in ids]
    return '<br/>'.join(links)

  def instancelink(self, instance):
    p = markup.page()
    p.a(instance,href="text/%s"%instance)
    return str(p)
  
  @cherrypy.expose
  def classmaps(self, name=None):
    if name is None:
      classmaps = self.ds.classmap_names
      links = ["<a href=classmaps?name=%s>%s</a>"%(i,i) for i in classmaps]
      return '<br/>'.join(links)
    else:
      page = markup.page()
      page.init()
      page.ul()
      for klass in sorted(self.class_index[name]):
        page.li(klass)
        page.ul()
        for instance in sorted(self.class_index[name][klass]):
          page.li(self.instancelink(instance))
        page.ul.close()
      page.ul.close()
      return str(page)

  @cherrypy.expose
  def text(self, instance):
    raw_text = self._text[instance]
    return pretty(raw_text)


if __name__ == "__main__":
  dsname = sys.argv[1]
  
  sys.path.append('.')
  try:
    exec('from %s import %s as ds' % tuple(dsname.rsplit('.',1)))
  except ImportError, e:
    try:
      exec('from hydrat.corpora.%s import %s as ds' % tuple(dsname.rsplit('.',1)))
    except ImportError, e:
      logger.debug(e)
      print("Unable to locate %s" % dsname)
      sys.exit(-1)
    except TypeError, e:
      logger.debug(e)
      print("%s is not a dataset" % dsname)
      sys.exit(-1)

  cherrypy.quickstart(TextDatasetBrowser(ds()))

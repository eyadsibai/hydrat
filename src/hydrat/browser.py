"""
Hydrat interactive store browser based on CherryPy
"""
import cherrypy
import urllib
import StringIO
import hydrat.common.markup as markup
from hydrat.store import Store

#TODO: Serve jquery & css from a location packaged with the module.
CSS_URL = "http://hum.cs.mu.oz.au/~mlui/lib/blue_style/style.css"
def navbar():
  page = markup.page()
  with page.table:
    with page.tr:
      with page.td: page.a('spaces',    href='/spaces')
      with page.td: page.a('datasets',  href='/datasets')
      with page.td: page.a('tasks',     href='/tasks')
      with page.td: page.a('results',   href='/results')
  return str(page)

page_config=\
  { 'css'      : CSS_URL
  , 'charset'  : 'utf8'
  , 'header'   : navbar()
  }

def list_of_links(assoc_list):
  page = markup.page()
  page.ul()
  for k,v in assoc_list:
    page.li()
    page.a(k, href=v) 
    page.li.close()
  page.ul.close()
  return str(page)

def list_as_html(l):
  page = markup.page()
  page.ul()
  for text in l:
    page.li()
    page.add(text)
    page.li.close()
  page.ul.close()
  return str(page)

def dict_as_html(d):
  page = markup.page()
  page.table()
  for key in d:
    page.tr()
    page.td(key)
    page.td(str(d[key]))
    page.tr.close()
  page.table.close()
  return str(page)

from hydrat.result.interpreter import SingleHighestValue
from hydrat.display.summary_fns import sf_basic
from hydrat.display.html import TableSort 
class Results(object):
  def __init__(self, store):
    self.store = store
    # TODO: Parametrize these somehow!!
    self.summary_fn = sf_basic
    self.interpreter = SingleHighestValue()

  @cherrypy.expose
  def index(self):
    return self.list()

  @cherrypy.expose
  def list(self):
    from hydrat.display.tsr import result_summary_table, default_relevant
    page = markup.page()
    page.init(**page_config)

    summaries = []
    for uuid in self.store._resolve_TaskSetResults({}):
      result = self.store._get_TaskSetResult(uuid)
      summary = self.summary_fn(result, self.interpreter)
      link = markup.oneliner.a('link', href='view?'+urllib.urlencode({'uuid':uuid}))
      summary['link'] = str(link)
      link = markup.oneliner.a('link', href='matrix?'+urllib.urlencode({'uuid':uuid}))
      summary['pairs'] = str(link)
      summaries.append(summary)

    text = StringIO.StringIO()
    relevant = default_relevant[:]
    relevant.append(("Pairs", 'pairs'))
    with TableSort(text) as renderer:
      result_summary_table(summaries, renderer, relevant)

    page.add(text.getvalue())
    return str(page)

  @cherrypy.expose
  def view(self, uuid):
    from hydrat.display.tsr import render_TaskSetResult
    page = markup.page()
    page.init(**page_config)
    
    result = self.store._get_TaskSetResult(uuid)
    class_space = self.store.get_Space(result.metadata['class_space'])
    summary = self.summary_fn(result, self.interpreter)
    text = StringIO.StringIO()
    with TableSort(text) as result_renderer:
      render_TaskSetResult(result_renderer, result, class_space, self.interpreter, summary)

    page.add(text.getvalue())
    return str(page)

  @cherrypy.expose
  def matrix(self, uuid):
    from display.html import HTMLWriter
    result = self.store._get_TaskSetResult(uuid)
    class_space = self.store.get_Space(result.metadata['class_space'])
    matrix = result.overall_classification_matrix(self.interpreter)

    page = markup.page()
    page.init(**page_config)
    with page.table:
      with page.tr:
        page.th()
        [ page.th(c) for c in class_space ]
      for i, row in enumerate(matrix):
        with page.tr:
          page.th(class_space[i])
          for j, val in enumerate(row):
            with page.td:
              gs = class_space[i]
              cl = class_space[j]
              link = 'classpair?'+urllib.urlencode({'uuid':uuid, 'gs':gs, 'cl':cl})
              page.a(str(val), href=link)

    return str(page)

  @cherrypy.expose
  def classpair(self, uuid, gs, cl):
    from display.html import HTMLWriter
    result = self.store._get_TaskSetResult(uuid)
    class_space = list(self.store.get_Space(result.metadata['class_space']))
    # TODO: handle dataset vs eval_dataset. Should everything have an eval_dataset?
    if 'eval_dataset' in result.metadata:
      dataset = result.metadata['eval_dataset']
    else:
      dataset = result.metadata['dataset']
    docids = list(self.store.get_InstanceIds(dataset))
    pairs = result.overall_classpairs(self.interpreter)

    gs_i = class_space.index(gs)
    cl_i = class_space.index(cl)

    page = markup.page()
    page.init(**page_config)
    page.h1("Classified from '%s' to '%s'" % (gs,cl))
    key = (gs_i, cl_i)
    with page.ul:
      for i in pairs[key]:
        with page.li:
          # TODO: Link to the instance details
          id = docids[i]
          page.a(id, href='../datasets/%s/instances/%s' % (dataset, id))
    return str(page)

class Datasets(object):
  def __init__(self, store):
    self.store = store
    for dsname in self.store.list_Datasets():
      setattr(self, dsname, Dataset(store,dsname) )

  @cherrypy.expose
  def index(self):
    page = markup.page()
    page.init(**page_config)
    page.ul()
    for dsname in self.store.list_Datasets():
      page.li()
      page.a(dsname, href=dsname)
      page.li.close()
    page.ul.close()
    return str(page)

# TODO: Clean up and elaborate the dataset display code
class Dataset(object):
  def __init__(self, store, name):
    self.store = store
    self.name = name

  @property
  def classspaces(self):
    return self.store.list_ClassSpaces(self.name)

  @property
  def featurespaces(self):
    return self.store.list_FeatureSpaces(self.name)

  @property
  def tokenstreams(self):
    return self.store.list_TokenStreams(self.name)

  @property
  def instanceids(self):
    return self.store.get_InstanceIds(self.name)

  @cherrypy.expose
  def index(self):
    page = markup.page()
    page.init(**page_config)
    page.h1("Dataset %s" % self.name)
    page.a("Instances", href='instances')
    page.h2("TODO:Summary")
    page.h2("Class Spaces")
    page.ul()
    for class_space in sorted(self.classspaces):
      page.li(class_space)
    page.ul.close()
    page.h2("Feature Spaces")
    page.ul()
    for feature_space in sorted(self.featurespaces):
      page.li()
      page.a(feature_space, href="featurespace/%s" % feature_space)
      page.li.close()
    page.ul.close()
    page.h2("TokenStreams")
    page.ul()
    for tokenstream in sorted(self.tokenstreams):
      page.li()
      page.add(tokenstream)
      page.li.close()
    page.ul.close()
    return str(page)

  @cherrypy.expose
  def instances(self, id=None):
    page = markup.page()
    page.init(**page_config)
    if id is None:
      page.ul()
      for i in self.instanceids: 
        page.li(i)
      page.ul.close()
    else:
      page.h1(id)
      page.h2("TokenStreams")
      with page.ul:
        for ts in self.tokenstreams:
          with page.li:
            link = '../tokenstream/%s/%s' % (ts, id)
            page.a(ts, href=link)
      page.h2("Feature Spaces")
      with page.ul:
        for fs in self.featurespaces:
          with page.li:
            link = '../features/%s/%s' % (fs, id)
            page.a(fs, href=link)
      page.h2("Class Spaces")
      page.add(list_as_html(self.classspaces))
    return str(page)

  @cherrypy.expose
  def tokenstream(self, name, instance):
    page = markup.page()
    #TODO: should we really be specifying this here?
    page.init(**page_config)
    ts = self.store.get_TokenStreams(self.name, name)
    index = self.instanceids.index(instance)
    page.add(ts[index])
    return str(page)

  @cherrypy.expose
  def features(self, feature_space, instance_id):
    featuremap = self.store.get_FeatureMap(self.name, feature_space)
    instance_index = self.store.get_InstanceIds(self.name).index(instance_id) 
    instance = featuremap.raw[instance_index]
    present_features = instance.nonzero()[1]
    feature_space = self.store.get_Space(feature_space)

    page = markup.page()
    page.init(**page_config)
    page.table()
    for i in present_features:
      page.tr()
      page.td(feature_space[i])
      page.td(str(instance[0,i]))
      page.tr.close()
    page.table.close()
    return str(page)

  @cherrypy.expose
  def featurespace(self, name):
    tag = self.store.resolve_Space({'name':name})
    featuremap = self.store.get_FeatureMap(self.name, name)

    md = dict_as_html(featuremap.metadata)
    page = markup.page()
    page.init(**page_config)
    page.p(md)
    page.ul()
    for i in self.store.get_InstanceIds(self.name):
      page.li()
      page.a(i,href='../instance_features/%s/%s' % (i, name))
      page.li.close()
    page.ul.close()
    return str(page)




class StoreBrowser(object):
  def __init__(self, store):
    self.store = store
    self.results = Results(store)
    self.datasets = Datasets(store)

  @cherrypy.expose
  def index(self):
    links =\
      [ ( 'Spaces', 'spaces')
      , ( 'Datasets', 'datasets')
      , ( 'Tasks', 'tasks')
      , ( 'Results', 'results')
      ]
    page = markup.page()
    page.init(**page_config)
    page.add(list_of_links(links))
    return str(page)

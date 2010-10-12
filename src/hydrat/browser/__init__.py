"""
Hydrat interactive store browser based on CherryPy
"""
import cherrypy
import urllib
import StringIO
import hydrat.common.markup as markup
from hydrat.store import Store
from hydrat.display.html import TableSort, HTMLWriter

#TODO: Serve jquery & css from a location packaged with the module.
#TODO: Allow for customization of the summary function and/or display headers
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
  for key in sorted(d):
    page.tr()
    page.td(key)
    page.td(str(d[key]))
    page.tr.close()
  page.table.close()
  return str(page)

class Tasks(object):
  def __init__(self, store, bconfig):
    self.store = store
    self.summary_fn = bconfig.summary_fn
    self.interpreter = bconfig.interpreter
    self.relevant = bconfig.relevant

  @cherrypy.expose
  def index(self):
    page = markup.page()
    page.init(**page_config)
    with page.ul:
      for uuid in self.store._resolve_TaskSet({}):
        page.li(uuid)
      
    return str(page)

class Results(object):
  def __init__(self, store, bconfig):
    self.store = store
    self.summary_fn = bconfig.summary_fn
    self.interpreter = bconfig.interpreter
    self.relevant = bconfig.relevant

  @cherrypy.expose
  def index(self):
    return self.list()

  @cherrypy.expose
  def list(self):
    from hydrat.display.tsr import result_summary_table
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
    relevant = self.relevant[:]
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
          id = docids[i]
          # TODO: Build a table of what is available instead, for easy acces. Add mouseover ajax maybe.
          page.a(id, href='../datasets/%s/instances/%s' % (dataset, id))
          page.a('byte',href='../datasets/%s/tokenstream/byte/%s' % (dataset, id))
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
    page = markup.page()
    page.init(**page_config)

    tag = self.store.resolve_Space({'name':name})
    featuremap = self.store.get_FeatureMap(self.name, name)

    from hydrat.display.sparklines import histogram
    md = featuremap.metadata
    md['num_docs'] = featuremap.raw.shape[0]
    md['num_features'] = featuremap.raw.shape[1]
    feat_dist = featuremap.raw.sum(axis=0)
    md['Feature Occurrance distribution'] = markup.oneliner.img(src=histogram(feat_dist))
    md['Feature Occurrance mean'] = feat_dist.mean()
    md['Feature Occurrance std']  = feat_dist.std()
    doc_sizes = featuremap.raw.sum(axis=1)
    md['Document Size distribution'] = markup.oneliner.img(src=histogram(doc_sizes))
    md['Document Size mean'] = doc_sizes.mean()
    md['Document Size std']  = doc_sizes.std()
    page.add(dict_as_html(md))

    text = StringIO.StringIO()
    rows = []
    for i, id in enumerate(self.store.get_InstanceIds(self.name)):
      row = {}
      row['index'] = i
      row['id'] = markup.oneliner.a(id,href='../features/%s/%s' % (name, id))
      row['size'] = doc_sizes[i,0]
      row['bytes'] = markup.oneliner.a('link',href='../tokenstream/byte/%s' % id)
      rows.append(row)

    with TableSort(text) as renderer:
      renderer.dict_table( rows
                         , ['index', 'id', 'size', 'bytes'] 
                         , col_headings = ['Index', 'Identifier', 'Size', 'bytes']
                         )
    page.add(text.getvalue())
    return str(page)

class Spaces(object):
  def __init__(self, store, bconfig):
    self.store = store
    self.bconfig = bconfig

  @cherrypy.expose
  def index(self):
    page = markup.page()
    page.init(**page_config)

    cols = ['name', 'size', 'encoding']
    headings = ['Name', {'label':'Size', 'sorter':'digit'}, 'Encoding']

    page.h1('Feature Spaces')
    rows = [self.store.get_SpaceMetadata(s) for s in self.store.list_FeatureSpaces()]
    for r in rows:
      r['name'] = markup.oneliner.a(r['name'], href='view?' +urllib.urlencode({'name':r['name']}))
    text = StringIO.StringIO()
    with TableSort(text) as renderer:
      renderer.dict_table( rows
                         , cols
                         , col_headings = headings
                         #, title = 'Feature Spaces'
                         )
    page.add(text.getvalue())

    page.h1('Class Spaces')
    rows = [self.store.get_SpaceMetadata(s) for s in self.store.list_ClassSpaces()]
    for r in rows:
      r['name'] = markup.oneliner.a(r['name'], href='view?' +urllib.urlencode({'name':r['name']}))
    text = StringIO.StringIO()
    with TableSort(text) as renderer:
      renderer.dict_table( rows
                         , cols
                         , col_headings = headings
                         #, title = 'Feature Spaces'
                         )
    page.add(text.getvalue())
    return str(page)

  @cherrypy.expose
  def view(self, name):
    page = markup.page()
    page.init(**page_config)
    with page.ul:
      for item in self.store.get_Space(name):
        with page.li:
          page.add(str(item))
    return str(page)



class StoreBrowser(object):
  def __init__(self, store, bconfig):
    self.store = store
    self.results = Results(store, bconfig)
    self.datasets = Datasets(store)
    self.spaces = Spaces(store, bconfig)
    self.tasks = Tasks(store, bconfig)

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

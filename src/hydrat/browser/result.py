import cherrypy
import urllib
import StringIO
import hydrat.common.markup as markup
from hydrat.display.html import TableSort
from common import page_config
from display import list_as_html, dict_as_html, list_of_links

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


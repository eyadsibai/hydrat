"""
Hydrat interactive store browser based on CherryPy
"""
import cherrypy
import urllib
import StringIO
import hydrat.common.markup as markup
from hydrat.store import Store

#TODO: Ship this in the browser module!
CSS_URL = "http://hum.cs.mu.oz.au/~mlui/lib/blue_style/style.css"

def list_of_links(assoc_list):
  page = markup.page()
  page.ul()
  for k,v in assoc_list:
    page.li()
    page.a(k, href=v) 
    page.li.close()
  page.ul.close()
  return str(page)

def to_html(l):
  page = markup.page()
  page.ul()
  for text in l:
    page.li()
    page.add(text)
    page.li.close()
  page.ul.close()
  return str(page)

class Results(object):
  def __init__(self, store):
    self.store = store

  @cherrypy.expose
  def index(self):
    return self.list()

  @cherrypy.expose
  def list(self):
    from hydrat.result.interpreter import SingleHighestValue
    from hydrat.display.summary_fns import sf_basic
    from hydrat.display.html import TableSort 
    from hydrat.display.tsr import result_summary_table
    page = markup.page()
    page.init(css=CSS_URL)

    interpreter = SingleHighestValue()
    summaries = []
    for resname in self.store._resolve_TaskSetResults({}):
      result = self.store._get_TaskSetResult(resname)
      summary = sf_basic(result, interpreter)
      link = markup.page()
      link.a('link', href='view?'+urllib.urlencode({'uuid':resname}))
      summary['link'] = str(link)
      summaries.append(summary)

    text = StringIO.StringIO()
    with TableSort(text) as renderer:
      result_summary_table(summaries, renderer)

    page.add(text.getvalue())
    return str(page)

  @cherrypy.expose
  def view(self, uuid):
    from hydrat.result.interpreter import SingleHighestValue
    from hydrat.display.tsr import render_TaskSetResult
    from hydrat.display.summary_fns import sf_basic
    from hydrat.display.html import TableSort 
    page = markup.page()
    page.init(css=CSS_URL)
    
    result = self.store._get_TaskSetResult(uuid)
    class_space = self.store.get_Space(result.metadata['class_space'])
    interpreter = SingleHighestValue()
    summary = sf_basic(result, interpreter)
    text = StringIO.StringIO()
    with TableSort(text) as result_renderer:
      render_TaskSetResult(result_renderer, result, class_space, interpreter, summary)

    page.add(text.getvalue())
    return str(page)


class StoreBrowser(object):
  def __init__(self, store):
    self.store = store
    self.results = Results(store)

  @cherrypy.expose
  def index(self):
    links =\
      [ ( 'Spaces', 'spaces')
      , ( 'Datasets', 'datasets')
      , ( 'Tasks', 'tasks')
      , ( 'Results', 'results')
      ]
    page = markup.page()
    page.init(css=CSS_URL)
    page.add(list_of_links(links))
    return str(page)

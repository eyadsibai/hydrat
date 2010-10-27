import cherrypy
import urllib
import StringIO
import hydrat.common.markup as markup
from hydrat.display.html import TableSort
from common import page_config
from display import list_as_html, dict_as_html, list_of_links
from hydrat.display.tsr import result_summary_table
from hydrat.common import as_set


class Tasks(object):
  def __init__(self, store, bconfig):
    self.store = store
    self.summary_fn = bconfig.summary_fn
    self.interpreter = bconfig.interpreter
    self.relevant = bconfig.relevant

  @cherrypy.expose
  def index(self):
    return self.list()

  @cherrypy.expose
  def list(self, **params):
    page = markup.page()
    page.init(**page_config)

    summaries = []
    uuids = self.store._resolve_TaskSet(params)
    for uuid in uuids:
      summary = self.store._get_TaskSetMetadata(uuid)
      summary['_select'] = markup.oneliner.input(type='checkbox', name='uuid', value=uuid)
      summary['uuid'] = str(summary['uuid'])
      summaries.append(summary)

    page.h3('Parameters')
    page.add(dict_as_html(params))

    if len(summaries) == 0:
      page.h1('No tasksets for given parameters')
    else:
      relevant = [(k.title(),k) for k in sorted(summaries[0].keys()) if not k.startswith('_')]
      relevant.insert(0, ("Select", '_select'))

      text = StringIO.StringIO()

      with TableSort(text) as renderer:
        result_summary_table(summaries, renderer, relevant)

      page.p('Displaying %d tasksets' % len(uuids))

      page.form(action='receive', method='post')
      page.input(type='submit', name='action', value='view')
      if self.store.mode == 'a':
        page.input(type='submit', name='action', value='delete')
      page.br()
      page.add(text.getvalue())
      page.form.close()

    return str(page)

  @cherrypy.expose
  def receive(self, **params):
    if 'action' in params:
      action = params['action']
      del params['action']
      raise cherrypy.HTTPRedirect(action+'?'+urllib.urlencode(params, True))
    else:
      raise cherrypy.HTTPRedirect("list")

  @cherrypy.expose
  def view(self, uuid):
    uuid = as_set(uuid).pop()
    taskset = self.store._get_TaskSet(uuid)

    summaries = []
    for task in taskset.tasks:
      summary = {}
      summary['features']     = task.train_vectors.shape[1]
      summary['train_count']  = task.train_vectors.shape[0]
      summary['test_count']   = task.test_vectors.shape[0]
      summaries.append(summary)

    text = StringIO.StringIO()
    with TableSort(text) as renderer:
      result_summary_table(summaries, renderer)

    page = markup.page()
    page.init(**page_config)
    page.add(dict_as_html(taskset.metadata))
    page.add(text.getvalue())

    return str(page)

  @cherrypy.expose
  def delete(self, uuid, confirmed='N'):
    if self.store.mode != 'a':
      raise ValueError, "Store open in read-only mode"
    uuid = as_set(uuid)

    page = markup.page()
    page.init(**page_config)
    if confirmed == 'N':
      page.add("Delete the following taskset(s)?")
      with page.ul:
        for id in uuid: page.li(uuid)
      page.a('YES', href='delete?' + urllib.urlencode({'uuid':uuid, 'confirmed':'Y'}, True))
    else:
      page.add("Deleted the following results:")
      with page.ul:
        for id in uuid: 
          try:
            self.store._del_TaskSet(id)
            page.li('(Success) '+id)
          except KeyError:
            page.li('(Failure) '+id)
    return str(page)

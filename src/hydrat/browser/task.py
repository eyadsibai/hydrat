import cherrypy
import urllib
import StringIO
from collections import defaultdict

import hydrat.display.markup as markup
from hydrat.display.html import TableSort
from hydrat.display.tsr import result_summary_table
from hydrat.common import as_set

from common import page_config
from display import list_as_html, dict_as_html, list_of_links


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
    tasksets = self.store.get_TaskSets(params)
    for taskset in tasksets:
      summary = dict(taskset.metadata)
      summary['_select'] = markup.oneliner.input(type='checkbox', name='uuid', value=taskset.uuid)
      summary['uuid'] = taskset.uuid
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

      page.p('Displaying %d tasksets' % len(tasksets))

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

    page = markup.page()
    page.init(**page_config)
    page.add(dict_as_html(taskset.metadata))

    for i, task in enumerate(taskset.tasks):
      page.h2('Task')
      with page.ul:
        for key in task.weights.keys():
          page.a(key, href='./weight/%s/%s'% (key,uuid))

    return str(page)

  @cherrypy.expose
  def weight(self, weight_key, uuid):
    uuid = as_set(uuid).pop()
    taskset = self.store._get_TaskSet(uuid)
    tasks = taskset.tasks

    # TODO: Better space resolution. Should make this part of the taskset.
    space = self.store.get_Space(taskset.metadata['feature_desc'][0])

    weights = defaultdict(dict)
    cols = ['feature']
    for task_index, task in enumerate(tasks):
      task_id = 'fold%d' % task_index
      cols.append(task_id)
      weight = task.weights[weight_key]
      for i, w in enumerate(weight):
        weights[space[i]][task_id] = w

    rows = []
    for f in weights:
      d = weights[f]
      d['feature'] = f
      rows.append(d)

    page = markup.page()
    page.init(**page_config)
    page.add(dict_as_html(taskset.metadata))
    page.h2('%s on %s' % (weight_key, taskset.metadata['feature_desc']))

    page.dict_table(rows, cols, col_headings=cols)

    return page()

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

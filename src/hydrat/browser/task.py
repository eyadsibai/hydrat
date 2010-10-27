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
      #TODO: Add delete link
      summary['_delete'] = markup.oneliner.a('delete', href='delete?'+urllib.urlencode({'uuid':uuid}))
      summary['uuid'] = str(summary['uuid'])
      summaries.append(summary)

    page.h3('Parameters')
    page.add(dict_as_html(params))

    if len(summaries) == 0:
      page.h1('No tasksets for given parameters')
    else:
      relevant = [(k.title(),k) for k in sorted(summaries[0].keys()) if not k.startswith('_')]

      text = StringIO.StringIO()
      if self.store.mode == 'a':
        relevant = [("Delete", '_delete')] + relevant

      with TableSort(text) as renderer:
        result_summary_table(summaries, renderer, relevant)

      page.p('Displaying %d tasksets' % len(uuids))
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

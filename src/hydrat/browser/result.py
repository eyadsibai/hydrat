import cherrypy
import urllib
import numpy
import StringIO
import hydrat.common.markup as markup
from hydrat.display.html import TableSort
from common import page_config
from display import list_as_html, dict_as_html, list_of_links
from hydrat.common import as_set
from collections import defaultdict
from hydrat.display.summary_fns import result_metadata
from hydrat.display.tsr import result_summary_table


def results_metadata_map(store, params, max_uniq = 10):
  mapping = defaultdict(set)
  uuids = store._resolve_TaskSetResults(params)
  for uuid in uuids:
    result = store._get_TaskSetResult(uuid)
    summary = result.metadata
    for key in summary:
      try:
        mapping[key].add(str(summary[key]))
      except TypeError:
        # Skip unhashable values
        pass
  for key in mapping.keys():
    if len(mapping[key]) > max_uniq or len(mapping[key]) <= 1:
      del mapping[key]
  return mapping

class Results(object):
  def __init__(self, store, bconfig):
    self.store = store
    self.summary_fn = bconfig.summary_fn
    self.interpreter = bconfig.interpreter
    self.relevant = bconfig.relevant

  @cherrypy.expose
  def index(self):
    return self.list()


  def result_summary_page(self, params, page, summary_fn, relevant = None):
    summaries = []
    uuids = self.store._resolve_TaskSetResults(params)
    for uuid in uuids:
      result = self.store._get_TaskSetResult(uuid)
      summary = summary_fn(result, self.interpreter)
      #TODO: Pull these key-adders out
      link = markup.oneliner.a('link', href='view?'+urllib.urlencode({'uuid':uuid}))
      summary['link'] = str(link)
      link = markup.oneliner.a('link', href='matrix?'+urllib.urlencode({'uuid':uuid}))
      summary['pairs'] = str(link)
      summary['select'] = markup.oneliner.input(type='checkbox', name='uuid', value=uuid)
      if self.store.mode == 'a':
        link = markup.oneliner.a('delete', href='delete?'+urllib.urlencode({'uuid':uuid}))
        summary['delete'] = str(link)
      summaries.append(summary)

    page.h3('Parameters')
    page.add(dict_as_html(params))

    if len(summaries) == 0:
      page.h1('No results')
    else:
      # Show all our metadata if no filter is specified
      if relevant is None:
        relevant = [(k.title(),k) for k in sorted(summaries[0].keys())]

      text = StringIO.StringIO()
      relevant = [({'label':'Select','sorter':None},'select')] + relevant
      relevant.append(("Pairs", 'pairs'))
      if self.store.mode == 'a':
        relevant.append(("Delete", 'delete'))

      with TableSort(text) as renderer:
        result_summary_table(summaries, renderer, relevant)

      page.p('Displaying %d results' % len(uuids))
      page.form(action='compare', method='get')
      page.input(type='submit', value='Compare Selected')
      page.br()
      page.add(text.getvalue())
      page.form.close()


  @cherrypy.expose
  def list(self, **params):
    # TODO: Find a way of handling this that does not need feature_desc transform to be hardcoded
    if 'feature_desc' in params: params['feature_desc'] = tuple(sorted(as_set(params['feature_desc'])))
    page = markup.page()
    page.init(**page_config)

    # Show contstraint options
    mapping = results_metadata_map(self.store, params)
    param_links = {}
    for key in mapping:
      links = []
      values = mapping[key]
      new_params = dict(params)
      for value in values:
        new_params[key] = value
        links.append( markup.oneliner.a(value, href='list?'+urllib.urlencode(new_params, True)))
      param_links[key] = links
      
    page.add(dict_as_html(param_links))

    # Link to detailed results for these parameters
    with page.p:
      #TODO: This is how we can parametrize which key-value pairs to show
      page.a('Show detailed results', href='details?'+urllib.urlencode(params))

    # Draw the actual summary
    self.result_summary_page(params, page, result_metadata, None)
    return str(page)

  @cherrypy.expose
  def details(self, **params):
    if 'feature_desc' in params: params['feature_desc'] = tuple(sorted(as_set(params['feature_desc'])))
    page = markup.page()
    page.init(**page_config)
    self.result_summary_page(params, page, self.summary_fn, self.relevant)
    return str(page)

  @cherrypy.expose
  def compare(self, uuid):
    # TODO: Parametrize interpreter for non one-of-m highest-best results
    # TODO: Add a count of # of compared result which are correct
    # TODO: show metadata keys in which the results differ
    from hydrat.common import as_set
    from hydrat.result.interpreter import SingleHighestValue
    interpreter = SingleHighestValue()
    uuid = as_set(uuid)

    # Hardcode interpreter
    interpreter = SingleHighestValue()

    # Read results
    results = [ self.store._get_TaskSetResult(i) for i in uuid ]
    md = results[0].metadata
    ds_key = 'eval_dataset' if 'eval_dataset' in md else 'dataset'

    # Sanity check
    must_match = ['class_space',ds_key]
    for m in must_match:
      value_set = set(r.metadata[m] for r in results)
      if len(value_set) != 1:
        raise ValueError, "Non-uniform value for '%s' : %s" % (m, value_set)

    # Grab relevant data from store
    class_space = self.store.get_Space(md['class_space'])
    instance_ids = self.store.get_InstanceIds(md[ds_key])
    gs = self.store.get_ClassMap(md[ds_key], md['class_space']).raw

    # Build a mapping of uuid to interpreted cl output
    classifs = []
    for result in results:
      cl = result.overall_classification(range(len(instance_ids)))
      cl = cl.sum(axis=2)
      cl = interpreter(cl)
      classifs.append(cl)
    classifs = numpy.dstack(classifs)

    # Identify classes that neither GS nor CL utilize, we can just skip these.
    boring_cl_unused = numpy.logical_and(gs.sum(axis=0) == 0, classifs.sum(axis=2).sum(axis=0) == 0)
    int_cl = numpy.logical_not(boring_cl_unused)

    boring_inst_allright = (numpy.logical_and((classifs.sum(axis=2) == len(uuid)), gs).sum(axis=1) == 1)
    int_inst = numpy.logical_not(boring_inst_allright)

    clabels = numpy.array(class_space)[int_cl]
    instlabels = numpy.array(instance_ids)[int_inst]
    classifs = classifs[:,int_cl,:]
    classifs = classifs[int_inst,:,:]
    gs = gs[int_inst,:]
    gs = gs[:,int_cl]


    info = {}
    info['class_space']            = md['class_space']
    info[ds_key]                   = md[ds_key]
    info['Total Classes']          = len(class_space)
    info['Interesting Classes']    = len(clabels)
    info['Total Instances']        = len(instance_ids)
    info['Interesting Instances']  = len(instlabels)

    page = markup.page()
    page.init(**page_config)

    page.add(dict_as_html(info))

    with page.table:
      for key in sorted(reduce(set.union, (set(r.metadata.keys()) for r in results))):
        try:
          if len(set(r.metadata.get(key,'UNKNOWN') for r in results)) > 1:
            with page.tr:
              page.th(key)
              page.td()
              for r in results:
                page.td(str(r.metadata.get(key,'-')))
        except TypeError:
          continue

      with page.tr:
        page.th()
        page.th('Goldstandard')
        for id in uuid: page.th(id)

      for i, instance_id in enumerate(instlabels):
        with page.tr:
          with page.th:
            link = '../datasets/'+md[ds_key]+'/instances?'+urllib.urlencode({'id':instance_id})
            page.a(instance_id, href= link)

          inst_gs = gs[i]
          page.td(clabels[inst_gs])
          for j, r_id in enumerate(uuid):
            inst_cl = classifs[i,:,j]
            page.td(clabels[inst_cl], **{'class':'correct' if (inst_gs==inst_cl).all() else 'wrong'})
      
    return str(page)

  @cherrypy.expose
  def delete(self, uuid, confirmed='N'):
    if self.store.mode != 'a':
      raise ValueError, "Store open in read-only mode"
    uuid = as_set(uuid)

    page = markup.page()
    page.init(**page_config)
    if confirmed == 'N':
      page.add("Delete the following results?")
      with page.ul:
        for id in uuid: page.li(uuid)
      page.a('YES', href='delete?' + urllib.urlencode({'uuid':uuid, 'confirmed':'Y'}, True))
    else:
      page.add("Deleted the folliwng results:")
      with page.ul:
        for id in uuid: 
          try:
            self.store._del_TaskSetResult(id)
            page.li('(Success) '+id)
          except KeyError:
            page.li('(Failure) '+id)
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
    matrix_sans_diag = numpy.logical_not(numpy.diag(numpy.ones(len(class_space), dtype=bool))) * matrix
    interesting = numpy.logical_or(matrix_sans_diag.sum(axis=0), matrix_sans_diag.sum(axis=1))
    int_cs = numpy.array(class_space)[interesting]
    matrix = matrix[interesting].transpose()[interesting].transpose()

    page = markup.page()
    page.init(**page_config)
    with page.table:
      with page.tr:
        page.th()
        [ page.th(c) for c in int_cs ]
      for i, row in enumerate(matrix):
        with page.tr:
          page.th(int_cs[i])
          for j, val in enumerate(row):
            gs = int_cs[i]
            cl = int_cs[j]
            if val > 0 and gs != cl:
              td_attr={'class':'highlight'}
            else:
              td_attr={}
            page.td(**td_attr)
            link = 'classpair?'+urllib.urlencode({'uuid':uuid, 'gs':gs, 'cl':cl})
            page.a(str(val), href=link)
            page.td.close()

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


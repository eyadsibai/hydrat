import cherrypy
import urllib
import StringIO
import hydrat.common.markup as markup
from hydrat.display.html import TableSort
from common import page_config
from display import list_as_html, dict_as_html, list_of_links


from collections import defaultdict


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
  def list(self, **params):
    from hydrat.display.tsr import result_summary_table
    page = markup.page()
    page.init(**page_config)
    page.h3('Parameters')
    page.add(dict_as_html(params))

    summaries = []
    for uuid in self.store._resolve_TaskSetResults(params):
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
  def compare(self, uuid):
    # TODO: Parametrize interpreter for non one-of-m highest-best results
    # TODO: Add a count of # of compared result which are correct
    import numpy
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

    # TODO: Color the cells by correctness
    # TODO: Link to instance
    with page.table:
      with page.tr:
        page.th()
        page.th('Goldstandard')
        for id in uuid: page.th(id)

      for i, instance_id in enumerate(instlabels):
        with page.tr:
          page.th(instance_id)
          inst_gs = gs[i]
          page.td(clabels[inst_gs])
          for j, r_id in enumerate(uuid):
            inst_cl = classifs[i,:,j]
            page.td(clabels[inst_cl], **{'class':'correct' if (inst_gs==inst_cl).all() else 'wrong'})
      
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


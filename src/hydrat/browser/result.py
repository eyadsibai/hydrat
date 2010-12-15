import cherrypy
import urllib
import numpy
import StringIO
from collections import defaultdict

import hydrat.common.markup as markup
from hydrat.display.html import TableSort
from hydrat.common import as_set
from hydrat.display.tsr import result_summary_table, project_compound
from hydrat.result import classification_matrix
from hydrat.common.metadata import metamap

import hydrat.result.stats as stats
from display import list_as_html, dict_as_html, list_of_links
from common import page_config


KEY_SEP =':'

def results_metadata_map(store, params, max_uniq = 10):
  mapping = metamap( store._get_TaskSetResultMetadata(uuid) for uuid in store._resolve_TaskSetResults(params) )
  for key in mapping.keys():
    if len(mapping[key]) > max_uniq or len(mapping[key]) <= 1:
      del mapping[key]
  return mapping

from hydrat.summary import Summary
class Navigation(Summary):
  def init(self, result, interpreter):
    Summary.init(self, result, interpreter)
    self.uuid = str(result.metadata['uuid'])

  def key__link(self):
    link = markup.oneliner.a('link', href='view?'+urllib.urlencode({'uuid':self.uuid}))
    return str(link)
    
  def key__pairs(self):
    link = markup.oneliner.a('link', href='matrix?'+urllib.urlencode({'uuid':self.uuid}))
    return str(link)

  def key__select(self):
    link = markup.oneliner.input(type='checkbox', name='uuid', value=self.uuid)
    return str(link)

  # TODO: Offer this as an option associated with 'select' instead.
  def key__delete(self):
    link = markup.oneliner.a('delete', href='delete?'+urllib.urlencode({'uuid':self.uuid}))
    return str(link)

class Results(object):
  def __init__(self, store, bconfig):
    self.store = store
    self.summary_fn = bconfig.summary_fn
    self.interpreter = bconfig.interpreter
    self.relevant = bconfig.relevant

  @cherrypy.expose
  def index(self):
    raise cherrypy.HTTPRedirect('list')

  def result_summary_page(self, params, page, ext_summary_fn = None, relevant = None):
    uuids = self.store._resolve_TaskSetResults(params)
    int_id = self.interpreter.__name__

    summary_fn = Navigation()
    if ext_summary_fn is not None:
      summary_fn.extend(ext_summary_fn)

    summaries = self.get_full_summary(uuids, summary_fn)

    page.h3('Parameters')
    page.add(dict_as_html(params))

    if len(summaries) == 0:
      page.h1('No results')
    else:
      text = StringIO.StringIO()

      # Show all our metadata if no filter is specified
      if relevant is None:
        #TODO: hydrat.display.tsr.result_summary_table does this by default, need to refactor against that.
        relevant = [(k.title(),k) for k in sorted(summaries[0].keys()) if not k.startswith('_')]
      else:
        relevant = relevant[:]

      relevant.insert(0, ({'label':'Select','sorter':None},'_select') )
      relevant.append( ( {'sorter': None, 'label':"Details"}      , "_link"          ) )
      relevant.append(("Pairs", '_pairs'))
      if self.store.mode == 'a':
        relevant.insert(0,("Delete", '_delete'))

      with TableSort(text) as renderer:
        result_summary_table(summaries, renderer, relevant)

      page.p('Displaying %d results' % len(uuids))

      page.form(action='receive', method='post')
      page.input(type='submit', name='action', value='csv')
      page.input(type='submit', name='action', value='compare')
      page.input(type='submit', name='action', value='metamap')
      if self.store.mode == 'a':
        page.input(type='submit', name='action', value='delete')
      page.br()
      page.add(text.getvalue())
      page.form.close()

  @cherrypy.expose
  def receive(self, **params):
    if 'action' in params:
      action = params['action']
      del params['action']
      raise cherrypy.HTTPRedirect(action+'?'+urllib.urlencode(params, True))
    else:
      raise cherrypy.HTTPRedirect("list")

  @cherrypy.expose
  def list(self, **params):
    page = markup.page()
    page.init(**page_config)

    # Show contstraint options
    mapping = results_metadata_map(self.store, params)
    param_links = {}
    for key, values in mapping.iteritems():
      links = []
      new_params = dict(params)
      for v in values:
        label = '%s(%d)' % (v, values[v])
        if isinstance(v, str):
          new_params[key] = v
          links.append( markup.oneliner.a(label, href='list?'+urllib.urlencode(new_params, True)))
        else:
          links.append( label )
      param_links[key] = links
      
    page.add(dict_as_html(param_links))

    # Link to detailed results for these parameters
    with page.p:
      #TODO: This is how we can parametrize which key-value pairs to show
      page.a('Show detailed results', href='details?'+urllib.urlencode(params))

    # Draw the actual summary
    self.result_summary_page(params, page)
    return str(page)

  @cherrypy.expose
  def details(self, **params):
    page = markup.page()
    page.init(**page_config)
    self.result_summary_page(params, page, self.summary_fn, self.relevant)
    return str(page)

  @cherrypy.expose
  def compare(self, uuid, show_wrong='0', goldstandard=None):
    print "show_wrong", show_wrong, type(show_wrong)
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
    must_match = ['class_space']
    for m in must_match:
      value_set = set(r.metadata[m] for r in results)
      if len(value_set) != 1:
        raise ValueError, "Non-uniform value for '%s' : %s" % (m, value_set)
    # TODO: Check that the instance IDs match

    # Grab relevant data from store
    class_space = self.store.get_Space(md['class_space'])
    instance_ids = self.store.get_InstanceIds(md[ds_key])
    gs = self.store.get_ClassMap(md[ds_key], md['class_space']).raw

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
    boring_inst_allwrong = (numpy.logical_and((classifs.sum(axis=2) == len(uuid)), numpy.logical_not(gs)).sum(axis=1) == 1)
    if show_wrong != '0':
      boring_inst = boring_inst_allright 
    else:
      boring_inst = numpy.logical_or(boring_inst_allright, boring_inst_allwrong)
    int_inst = numpy.logical_not(boring_inst)


    # Keep only interesting instances and interesting classes
    clabels = numpy.array(class_space)[int_cl]
    instlabels = numpy.array(instance_ids)[int_inst]
    classifs = classifs[:,int_cl,:]
    classifs = classifs[int_inst,:,:]
    gs = gs[int_inst,:]
    gs = gs[:,int_cl]

    # Compute confusion pairs
    cm_all = []
    for i in xrange(classifs.shape[2]):
      cl = classifs[:,:,i]
      cm = classification_matrix(gs, cl)
      cm_all.append(cm)
    cm_all = numpy.dstack(cm_all)

    pairs = {}
    for i, l_i in enumerate(clabels):
      for j, l_j in enumerate(clabels):
        if i != j:
          pairs[(i,j)] = list(cm_all[i,j,:])

    pairs_by_size = sorted(pairs, key=lambda x: sum(pairs[x]), reverse=True)

    info = {}
    info['class_space']            = md['class_space']
    info[ds_key]                   = md[ds_key]
    info['Total Classes']          = len(class_space)
    info['Interesting Classes']    = len(clabels)
    info['Total Instances']        = len(instance_ids)
    info['Interesting Instances']  = len(instlabels)

    # Compute the set of keys present in the metadata over all results 
    all_keys = sorted(reduce(set.union, (set(r.metadata.keys()) for r in results)))
    # Compute the set of possible values for each key 
    values_set = {}
    for k in all_keys:
      for r in results:
        try:
          values_set[k] = set(r.metadata.get(k,'UNKNOWN') for r in results)
        except TypeError:
          # skip unhashable
          pass
    # Compute the set of key-values which all the results have in common
    common_values = dict( (k, values_set[k].pop()) for k in values_set if len(values_set[k]) == 1)

    page = markup.page()
    page.init(**page_config)

    # Show summary
    page.h2('Summary')
    page.add(dict_as_html(info))

    # Statistical Tests
    # TODO: Add measures of correlation as well
    page.h2('Statistical Significance')
    if len(uuid) == 1:
      page.p("No test for single result")
    elif len(uuid) == 2:
      #page.p("McNemar's test")
      mcnemar_result = stats.mcnemar(self.interpreter, results[0], results[1])
      page.add(dict_as_html(dict(mcnemar=mcnemar_result)))
      mcnemar_pc = stats.mcnemar(self.interpreter, results[0], results[1], perclass=True)[int_cl]
      if goldstandard is not None:
        gs_i = list(clabels).index(goldstandard)
        page.add(dict_as_html({goldstandard:mcnemar_pc[gs_i]}))
      else:
        page.add(dict_as_html(dict(zip(clabels, mcnemar_pc))))
    else:
      page.p("ANOVA")



    # Show common metadata
    page.h2('Common Metadata')
    page.add(dict_as_html(common_values))

    # Give the option to show/hide instances that are entirely wrong
    if show_wrong != '0':
      page.a('Hide Wrong', href='compare?' + urllib.urlencode({'uuid':uuid, 'show_wrong':0}, True))
    else:
      page.a('Show Wrong', href='compare?' + urllib.urlencode({'uuid':uuid, 'show_wrong':1}, True))

    # Confusion pairs tabulation
    with page.table:
      for key in values_set:
        # Display keys which differ
        if len(values_set[key]) > 1:
          with page.tr:
            page.th(key)
            page.td()
            for r in results:
              page.td(str(r.metadata.get(key,'-')))
      with page.tr:
        page.th()
        page.td()
        for id in uuid: page.th(id)

      # Display classification pairs
      for pair in pairs_by_size:
        if sum(pairs[pair]) == 0:
          # Can stop display, all the rest are zero
          break
        fr = clabels[pair[0]]  
        to = clabels[pair[1]] 
        if goldstandard is None or goldstandard == fr:
          with page.tr:
            label = ' => '.join((fr, to))
            page.th(label)
            page.td()
            for i, id in enumerate(uuid):
              with page.td:
                page.a(pairs[pair][i], href='classpair?'+urllib.urlencode({'gs':fr,'cl':to,'uuid':id}))

      # Display individual instances
      with page.tr:
        page.th()
        page.th('Goldstandard')
        for id in uuid: 
          with page.th:
            page.a(id, href='view?'+urllib.urlencode({'uuid':id}))

      for i, instance_id in enumerate(instlabels):
        inst_gs = gs[i]
        label = clabels[inst_gs]

        # Handle instance with no goldstandard labels
        if len(label) == 0: label = ''

        if goldstandard is None or goldstandard == label:
          with page.tr:
            with page.th:
              link = '../datasets/'+md[ds_key]+'/instances?'+urllib.urlencode({'id':instance_id})
              page.a(instance_id, href= link)

            with page.td:
              page.a(label, href='compare?' + urllib.urlencode({'uuid':uuid, 'show_wrong':show_wrong, 'goldstandard':label}, True))
            for j, r_id in enumerate(uuid):
              inst_cl = classifs[i,:,j]
              labels = list(clabels[inst_cl])
              page.td(labels, **{'class':'correct' if (inst_gs==inst_cl).all() else 'wrong'})
      
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

  def get_full_summary(self, uuids, summary_fn = None):
    # Build the display summaries as we go, based on the stored summaries and any additional
    # summary function supplied.
    int_id = self.interpreter.__name__
    if summary_fn is None:
      summary_fn = self.summary_fn
    summaries = []

    for uuid in uuids:
      summary = self.store.get_Summary(uuid, int_id)
      missing_keys = set(summary_fn.keys) - set(summary)
      if len(missing_keys) > 0:
        result = self.store._get_TaskSetResult(uuid)
        summary_fn.init(result, self.interpreter)
        new_values = dict( (key, summary_fn[key]) for key in missing_keys )
        summary.update(new_values)
      summaries.append(summary)
    return summaries


  @cherrypy.expose
  def csv(self, uuid, columns=None):
    # TODO: Let user select columns
    # TODO: Apply the summary function. The summaries we get are straight from the store, and don't
    #       have the modifications resulting from browser-config
    uuid = as_set(uuid)
    int_id = self.interpreter.__name__
    fieldnames = zip(*self.relevant)[1]
    rows = self.get_full_summary(uuid)
    rows = project_compound(rows, fieldnames)
    
    import csv
    from cStringIO import StringIO
    out = StringIO()
    writer = csv.DictWriter(out, fieldnames, extrasaction='ignore')
    writer.writerows(rows)
    text = out.getvalue()
    cherrypy.response.headers["Content-Type"] = "text/csv"
    cherrypy.response.headers["Content-Length"] = len(text)
    cherrypy.response.headers["Content-Disposition"] = "attachment; filename=hydrat_browser.csv"
    return text
      

  @cherrypy.expose
  def metamap(self, uuid):
    import cgi
    import pprint
    uuids = as_set(uuid)
    int_id = self.interpreter.__name__
    map = metamap(self.store.get_Summary(uuid, int_id) for uuid in uuids)
    page = markup.page()
    page.init(**page_config)
    page.pre(cgi.escape(pprint.pformat(list(uuids))))
    page.add(dict_as_html(map))
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
  def matrix(self, uuid, threshold=0):
    threshold = int(threshold)
    result = self.store._get_TaskSetResult(uuid)
    class_space = self.store.get_Space(result.metadata['class_space'])
    matrix = result.overall_classification_matrix(self.interpreter)
    matrix_sans_diag = numpy.logical_not(numpy.diag(numpy.ones(len(class_space), dtype=bool))) * matrix
    matrix_sans_diag *= matrix >= threshold
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
            if val > threshold and gs != cl:
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
    page.add(dict_as_html(result.metadata))
    page.h1("Classified from '%s' to '%s'" % (gs,cl))
    key = (gs_i, cl_i)
    tokenstreams = sorted(self.store.list_TokenStreams(dataset))
    featurespaces = sorted(self.store.list_FeatureSpaces(dataset))
    with page.table:
      for i in pairs[key]:
        with page.tr:
          id = docids[i]
          with page.th:
            page.a(id, href='../datasets/%s/instances?' % dataset + urllib.urlencode({'id':id}))
          for ts in tokenstreams:
            with page.td:
              page.a(ts,href='../datasets/%s/tokenstream/%s/%s' % (dataset, ts, id))
          for fs in featurespaces:
            with page.td:
              page.a(fs,href='../datasets/%s/features/%s/%s' % (dataset, fs, id))
    return str(page)


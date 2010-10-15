import cherrypy
import urllib
import StringIO
import hydrat.common.markup as markup
from hydrat.display.html import TableSort
from common import page_config
from display import list_as_html, dict_as_html, list_of_links, dict_table

class Datasets(object):
  def __init__(self, store):
    self.store = store
    for dsname in self.store.list_Datasets():
      setattr(self, dsname, Dataset(store,dsname) )

  @cherrypy.expose
  def class_distribution(self):
    # TODO: Sort the class distribution by class population across all datasets, so
    # we get a pretty descending graph.
    rows=[]
    for dsname in self.store.list_Datasets():
      class_spaces = self.store.list_ClassSpaces(dsname)
      row = {}
      row['name']            = markup.oneliner.a(dsname, href=dsname)
      for c in class_spaces:
        classmap = self.store.get_ClassMap(dsname, c)
        from hydrat.display.sparklines import barchart 
        dist = classmap.raw.sum(axis=0)
        width = min(10, 500 / len(dist))
        image = markup.oneliner.img(src=barchart(dist, height=15, width=width, gap=0)) 
        row[c] = markup.oneliner.a(image, href='%s/classspace/%s' % (dsname,c))
      rows.append(row)

    cols = [ ('Dataset Name', 'name') ]
    for c in self.store.list_ClassSpaces():
      cols.append( (c,c) )
    col_headings, col_keys = zip(*cols)

    page = markup.page()
    page.init(**page_config)
    table = dict_table(rows, col_keys, col_headings, default='ABSENT')
    page.add(table)
    return str(page)

  @cherrypy.expose
  def index(self):
    rows = []
    for dsname in self.store.list_Datasets():
      row = {}
      row['name']            = markup.oneliner.a(dsname, href=dsname)
      row['instances']       = str(len(self.store.get_InstanceIds(dsname)))
      row['feature_spaces']  = str(len(self.store.list_FeatureSpaces(dsname)))
      row['class_spaces']    = str(len(self.store.list_ClassSpaces(dsname)))
      row['tokenstreams']    = str(len(self.store.list_TokenStreams(dsname)))
      rows.append(row)

    cols = [ ('Dataset Name', 'name')
           , ('Instances', 'instances')
           , ('Feature Spaces', 'feature_spaces')
           , ('Class Spaces', 'class_spaces')
           , ('Tokenstreams', 'tokenstreams')
           ]
    col_headings, col_keys = zip(*cols)

    page = markup.page()
    page.init(**page_config)
    page.a('Class Distribution', href='class_distribution')
    table = dict_table(rows, col_keys, col_headings)
    page.add(table)
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
    page.h2("Summary")
    page.p("%d instances" % len(self.instanceids))

    page.h2("Class Spaces")
    with page.ul:
      for class_space in sorted(self.classspaces): 
        with page.li: page.a(class_space, href="classspace/%s" % class_space)

    page.h2("Feature Spaces")
    with page.ul:
      for feature_space in sorted(self.featurespaces):
        with page.li: page.a(feature_space, href="featurespace/%s" % feature_space)

    page.h2("TokenStreams")
    with page.ul:
      for tokenstream in sorted(self.tokenstreams):
        page.li(tokenstream)
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
  def classspace(self, name):
    space = self.store.get_Space(name)
    classmap = self.store.get_ClassMap(self.name, name)

    class_dist = dict(zip(space, classmap.raw.sum(axis=0)))

    page = markup.page()
    page.init(**page_config)
    page.h2('Class Distribution')
    # TODO: Do this as a tablesort
    page.add(dict_as_html(class_dist))
    return str(page)

  @cherrypy.expose
  def featurespace(self, name):
    page = markup.page()
    page.init(**page_config)

    featuremap = self.store.get_FeatureMap(self.name, name)

    from hydrat.display.sparklines import histogram
    md = featuremap.metadata
    md['num_docs'] = featuremap.raw.shape[0]
    md['num_features'] = featuremap.raw.shape[1]
    feat_dist = featuremap.raw.sum(axis=0)
    md['num_features_nonzero'] = feat_dist.nonzero()[1].shape[1]
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


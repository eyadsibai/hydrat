"""
Hydrat interactive store browser based on CherryPy
"""
import cherrypy
import hydrat.common.markup
from hydrat.store import Store


def dict_as_html(d):
  page = markup.page()
  page.table()
  for key in d:
    page.tr()
    page.td(key)
    page.td(str(d[key]))
    page.tr.close()
  page.table.close()
  return str(page)

class Space(object):
  def __init__(self, name, store):
    self.name = name
    self.store = store
    self._tag = store.resolve_Space({'name':name})
    self._metadata = self.store.get_Metadata(self._tag)
    self.space = self.store.get_Space(self._tag)

  @cherrypy.expose
  def index(self):
    page = markup.page()
    page.init()
    page.p("SPACE %s with %d members" % (self.name, len(self.space)))
    page.a("tag", href='tag')
    page.a("metadata", href='metadata')
    page.a("members", href='members')
    return str(page)

  @cherrypy.expose
  def tag(self):
    return self._tag

  @cherrypy.expose
  def metadata(self):
    return dict_as_html(self._metadata)

  @cherrypy.expose
  def members(self):
    encoding = self._metadata['encoding']

    page = markup.page()
    page.init()
    page.ul()
    for feat in self.space:
      if encoding == 'ascii':
        page.li(feat)
      else:
        page.li(feat.encode(encoding))
    page.ul.close()

    return str(page)


class Spaces(object):
  def __init__(self, store, space_lister):
    self.store = store
    self.space_lister = space_lister
    for space in space_lister():
      setattr(self, space, Space(space, store))

  @cherrypy.expose
  def index(self):
    page = markup.page()
    page.init()
    page.ul()
    for space in sorted(self.space_lister()):
      page.li()
      page.a(space, href=space)
      page.li.close()
    page.ul.close()
    return str(page)

class Datasets(object):
  def __init__(self, store):
    self.store = store
    for dsname in self.store.list_Datasets():
      setattr(self, dsname, Dataset(store,dsname) )

  @cherrypy.expose
  def index(self):
    page = markup.page()
    page.init()
    page.ul()
    for dsname in self.store.list_Datasets():
      page.li()
      page.a(dsname, href=dsname)
      page.li.close()
    page.ul.close()
    return str(page)

class Dataset(object):
  def __init__(self, store, name):
    self.store = store
    self.name = name
    self._tag = store.resolve_Dataset(name)
    self._featurespaces = store.list_FeatureSpaces(name)
    self._classspaces = store.list_ClassSpaces(name)

  @cherrypy.expose
  def index(self):
    page = markup.page()
    page.init()
    page.h1("Dataset %s" % self.name)
    page.h2("Class Spaces")
    page.ul()
    for class_space in sorted(self._classspaces):
      tag = self.store.resolve_Space({'name':class_space})
      page.li(class_space)
    page.ul.close()
    page.h2("Feature Spaces")
    page.ul()
    for feature_space in sorted(self._featurespaces):
      tag = self.store.resolve_Space({'name':feature_space})
      page.li()
      page.a(feature_space, href="featurespace/%s" % feature_space)
      page.li.close()
    page.ul.close()
    return str(page)

  @cherrypy.expose
  def instances(self):
    page = markup.page()
    page.init()
    page.ul()
    for i in self.store.instance_identifiers(self._tag):
      page.li(i)
    page.ul.close()
    return str(page)

  @cherrypy.expose
  def instance_features(self, instance_id, feature_space):
    tag = self.store.resolve_Space({'name':feature_space})
    featuremap = self.store.get_FeatureMap(self._tag, tag)
    instance_index = self.store.instance_identifiers(self._tag).index(instance_id) 
    instance = featuremap.raw[instance_index]
    present_features = instance.nonzero()[1]
    feature_space = self.store.get_Space(tag)

    page = markup.page()
    page.init()
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
    tag = self.store.resolve_Space({'name':name})
    featuremap = self.store.get_FeatureMap(self._tag, tag)

    md = dict_as_html(featuremap.metadata)
    page = markup.page()
    page.init()
    page.p(md)
    page.ul()
    for i in self.store.instance_identifiers(self._tag):
      page.li()
      page.a(i,href='../instance_features/%s/%s' % (i, name))
      page.li.close()
    page.ul.close()
    return str(page)

  @cherrypy.expose
  def tag(self):
    return self._tag

class StoreBrowser(object):
  def __init__(self, store):
    self.store = store
    self.datasets = Datasets(self.store)
    self.featurespaces = Spaces(self.store, self.store.list_FeatureSpaces)
    self.classspaces = Spaces(self.store, self.store.list_ClassSpaces)

  @cherrypy.expose
  def index(self):
    page = markup.page()
    page.init()
    page.ul()
    page.li()
    page.a('Feature Spaces', href='featurespaces')
    page.li.close()
    page.li()
    page.a('Class Spaces', href='classspaces')
    page.li.close()
    page.ul.close()

    return str(page)

if __name__ == "__main__":
  import sys
  store_path = sys.argv[1]
  store = UniversalStore(store_path)
  cherrypy.quickstart(StoreBrowser(store))

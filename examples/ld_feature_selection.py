from hydrat.corpora.dummy import dummy
from hydrat.proxy import DataProxy, TransductiveLOO
from hydrat.store import Store
from hydrat.transformer.featureselect import LangDomain
from hydrat.transformer import Transform

datasets = [
  dummy(10),
  dummy(20),
  dummy(30),
]


if __name__ == "__main__":
  store = Store.from_caller()
  proxies = []
  for ds in datasets:
    proxy = DataProxy(ds, store)
    proxy.feature_spaces = 'byte_unigram'
    proxy.class_space = 'dummy_default'
    proxies.append(proxy)
  
  proxy = TransductiveLOO(proxies)

  ts = store.new_TaskSet(proxy)
  tss = Transform(ts, LangDomain(proxy.domainmap))
  x = list(tss)

  import pdb;pdb.set_trace()

    



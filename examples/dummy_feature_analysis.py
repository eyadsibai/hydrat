import hydrat.frameworks.crossvalidation as exp
from hydrat.corpora.dummy import unicode_dummy

if __name__ == "__main__":
  exp.default_crossvalidation(unicode_dummy(), work_path='output')

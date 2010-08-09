from hydrat.display.store import write_report
from hydrat.store import DatasetStore
from hydrat.common.log import getHydraLogger
from hydrat import configuration as config
import sys

if __name__ == "__main__":
  assert 2 <= len(sys.argv) <= 3, "Usage: %s [output_directory] [model_file]" % sys.argv[0]
  logger = getHydraLogger()
  model_path = sys.argv[2] if len(sys.argv) == 3 else config.getpath('dataStore')
  report_path = sys.argv[1]
  store = DatasetStore(model_path)
  write_report(store, report_path)

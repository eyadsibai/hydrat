from hydrat.store import Store
from hydrat.summary import classification_summary, Metadata
from hydrat.result.interpreter import SingleHighestValue
import sys

if __name__ == "__main__":
  store = Store(sys.argv[1],'a')
  # TODO: Parametrize on summary_function and interpreter
  summary_fn = classification_summary
  summary_fn.extend(Metadata(['eval_dataset']))
  interpreter = SingleHighestValue()
  int_id = interpreter.__name__
  for tsr_id in store._resolve_TaskSetResults({}):
    summary = store.get_Summary(tsr_id, int_id)
    missing_keys = set(summary_fn.keys) - set(summary)
    if len(missing_keys) > 0:
      result = store._get_TaskSetResult(tsr_id)
      summary_fn.init(result, interpreter)
      new_values = dict( (key, summary_fn[key]) for key in missing_keys )
      store.add_Summary(tsr_id, int_id, new_values) 
    print "Added", missing_keys, "to", tsr_id

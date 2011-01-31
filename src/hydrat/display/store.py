from cStringIO import StringIO
from hydrat.display.tsr import result_summary_table
from hydrat.display.html import TableSort

def results2html(store, browser_config, tsr_metadata=None):
  if tsr_metadata is None:
    tsr_metadata = {}

  interpreter = browser_config.interpreter
  summary_fn  = browser_config.summary_fn
  int_id = interpreter.__name__

  summaries = []
  for uuid in store._resolve_TaskSetResults(tsr_metadata):
    summary = store.get_Summary(uuid, int_id)

    # TODO: refactor this against summary in frameworks.offline
    missing_keys = set(summary_fn.keys) - set(summary)
    if len(missing_keys) > 0:
      result = store._get_TaskSetResult(uuid)
      summary_fn.init(result, interpreter)
      new_values = dict( (key, summary_fn[key]) for key in missing_keys )
      summary.update(new_values)

    summaries.append(summary)

  io = StringIO()
  with TableSort(io) as renderer:
    result_summary_table(summaries, renderer, relevant = browser_config.relevant)
  return io.getvalue()

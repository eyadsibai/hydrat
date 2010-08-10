import progressbar as pb

def get_widget(label):
  return [ label, ' ', pb.Percentage(), ' ', pb.Bar(marker='>',left='[',right=']'), ' ', pb.ETA() ]

class ProgressBar(pb.ProgressBar):
  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.finish()


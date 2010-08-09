import Image, ImageDraw
import StringIO
import urllib
import numpy

# Based on http://bitworking.org/news/Sparklines_in_data_URIs_in_Python
def divergence_bar(results, baseline = None, height = 20, width = 10, gap = 2):
  # Use the mean as a baseline if none is provided.
  baseline = baseline if baseline is not None else float(sum(results)) / len(results)
  im = Image.new("RGBA", (len(results)*(width + gap), height), (0,0,0,0))
  draw = ImageDraw.Draw(im)
  limit = max(map(lambda x : abs( x - baseline ), results))
  scaling_factor = height / 2.0 / limit
  mid = int(round(height/2.0))
  for (r, i) in zip(results, range(0, len(results)*(width+gap), (width+gap))):
    diff = abs(r - baseline)
    prop_diff = diff * scaling_factor
    if r >= baseline:
      draw.rectangle((i, mid, i+width-1, mid - prop_diff ), fill='blue')
    else:
      draw.rectangle((i, mid, i+width-1, mid + prop_diff ), fill='red')
  del draw

  f = StringIO.StringIO()
  im.save(f, "PNG")
  return 'data:image/png,' + urllib.quote(f.getvalue())

def barchart(values, height = 20, width = 5, gap = 1, color='black'):
  limit = float(max(values))
  im = Image.new("RGBA", (len(values)*(width + gap), height), (0,0,0,0))
  draw = ImageDraw.Draw(im)
  scaling_factor = height / limit

  for (r, i) in zip(values, range(0, len(values)*(width+gap), (width+gap))):
    draw.rectangle( (i, height , i + width - 1, height - (r*scaling_factor) ) ,fill=color)
  del draw

  f = StringIO.StringIO()
  im.save(f, "PNG")
  return 'data:image/png,' + urllib.quote(f.getvalue())

def histogram(values, bins = 100, height = 20, width = 1, gap = 1, color = 'black'):
  hist,bounds = numpy.histogram(values, bins = bins, new=True)
  return barchart(hist, height, width, gap, color)

  

from textcat import TextCatConfig
from langdetect import LangDetectConfig

from hydrat.dataset.iso639 import ISO639_1_CODES

def goog2iso639_1(lang):
  if lang == 'fil':             return 'tl'
  if lang == 'zh-TW':           return 'zh'
  if lang == 'zh-CN':           return 'zh'
  if lang == 'iw':              return 'he'
  elif lang in ISO639_1_CODES:  return lang
  else:                         return 'UNKNOWN'


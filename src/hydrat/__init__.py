import os.path
from common.configuration import HydratConfiguration
if os.path.exists('config.ini'):
  configuration = HydratConfiguration('config.ini')
elif os.path.exists(os.path.expanduser('~/.hydrat/config.ini')):
  configuration = HydratConfiguration(os.path.expanduser('~/.hydrat/config.ini'))
else:
  raise ValueError, "Can't find a configuration file!"

def main():
  print "Hello world from hydrat"

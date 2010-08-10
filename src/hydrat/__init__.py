import sys

from optparse import OptionParser 

import configuration

config = configuration.read_configuration()

def main():
  parser = OptionParser()
  parser.add_option\
    ( "-c", "--config"
    , dest="config"
    , help="Read configuration from a file"
    , metavar="FILENAME"
    )
  parser.add_option\
    ( "--write_config", dest="write_config" 
    , help="Write default configuration to a file" 
    , metavar="FILENAME"
    )

  options, args = parser.parse_args()

  if options.write_config is not None:
    configuration.write_default_configuration(options.write_config)
    sys.exit(0)
  if options.config is not None:
    global config
    config = configuration.read_configuration(options.config)

  print config.get('paths', 'scratch')
  #parser.error('Try invoking this program with the --help option')
  #parser.usage()

if __name__ == "__main__":
  main()

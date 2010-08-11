from optparse import OptionParser 

import configuration
import cli

# Global configuration
config = configuration.read_configuration()
configuration.process_configuration(config)

# Global random number generator
rng = configuration.rng 


def main():
  parser = OptionParser()
  parser.add_option\
    ( "-c", "--config"
    , dest="config"
    , help="Read configuration from a file"
    , metavar="FILENAME"
    )

  options, args = parser.parse_args()

  if options.config is not None:
    global config
    config = configuration.read_configuration(options.config)
    configuration.process_configuration(config)

  # Handle commands
  cli.handle_commands(args)

if __name__ == "__main__":
  main()

import sys

class Framework:

  def __init__(self):
    pass

  def run(self, command_string):
    from NLP_LIB.nlp_core.engine import main
    command_arr = command_string.split(' ')
    main(command_arr)

# Main entry point
if __name__ == '__main__':
  Framework().run(' '.join(sys.argv))

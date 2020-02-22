class Framework:

  def __init__(self):
    pass

  def execute(self, command_string):
    from NLP_LIB.nlp_core.engine import main
    command_arr = command_string.split(' ')
    main(command_arr)


from flask import Flask, request
import socket
from threading import Thread
import keras
import json
from NLP_LIB.nlp_core.model_wrapper import ModelWrapper, SequenceModelWrapper, TrainableModelWrapper

app = Flask(__name__)
server = None

@app.route('/predict', methods=["GET","POST"])
def predict():
  global server

  trainer = server.trainer
  model = server.model
  graph = server.graph
  session = server.session

  if not isinstance(trainer.trainable_model, SequenceModelWrapper):
    try:
      text = None
      if request.method == "POST":		
        if 'text' in request.form:
          text = request.form['text']
      else:
        text = request.args.get('text')
      if text is None or len(text) == 0:
        return json.dumps({"status": "FAIL", "msg": "Empty input"})
      res = None
      with graph.as_default():
        keras.backend.set_session(session)
        Yt, Ys, YProbs = trainer.predict('predict', 'argmax', 0, 'str', text, prev_output_path = None, model = model)
        Ys = Ys.tolist()
        YProbs = YProbs.tolist()
      return json.dumps({"status": "OK", "y": Ys, "yt": Yt, "scores": YProbs}, ensure_ascii=False)
    except Exception as e:
      print('[ERROR]')
      print(e)
      return json.dumps({"status": "FAIL", "msg": str(e)})
  else:
    try:
      text = None
      count = None
      beam = None
      if request.method == "POST":		
        if 'text' in request.form:
          text = request.form['text']
        if 'count' in request.form:
          count = request.form['count']
        if 'beam' in request.form:
          beam = request.form['beam']
      else:
        text = request.args.get('text')
        count = request.args.get('count')
        beam = request.args.get('beam')
      if text is None or len(text) == 0:
        return json.dumps({"status": "FAIL", "msg": "Empty input"})
      if count is None or len(count) == 0:
        count = '10'
      try:
        count = int(count)
      except Exception as e:
        return json.dumps({"status": "FAIL", "msg": "Invalid count parameter"})      
      if beam is None or len(beam) == 0:
        beam = '3'
      try:
        beam = int(beam)
      except Exception as e:
        return json.dumps({"status": "FAIL", "msg": "Invalid beam parameter"})      
      res = None
      with graph.as_default():
        keras.backend.set_session(session)
        Yt, Ys, YProbs = trainer.predict('generate', 'beam' + str(beam), count, 'str', text, prev_output_path = None, model = model)
        Ys = Ys.tolist()
        #YProbs = YProbs.tolist()
      return json.dumps({"status": "OK", "y": Ys, "yt": Yt}, ensure_ascii=False)
    except Exception as e:
      print('[ERROR]')
      print(e)
      return json.dumps({"status": "FAIL", "msg": str(e)})

@app.route('/')
def render_test_page():
  model_name = server.model_name
  trainer = server.trainer  
  model = server.model
  graph = server.graph
  session = server.session
  example_api_endpoint = '/predict?text=TEXT'
  output_postprocess = """
                for (var i=0;i<res.yt.length;i++) {
                  if (i > 0)
                    output += ',';
                  output = output + res.yt[i] + ' (' + res.scores[i][res.y[i]] + ') ';
                }  
  """
  if isinstance(trainer.trainable_model, SequenceModelWrapper):  
    example_api_endpoint = '/predict?text=TEXT&count=10&beam=3'
    output_postprocess = """
      output = output + res.yt[0];
    """
  return """
    <html>
      <title>NLP_LIB API Test Page</title>
      <head>
        <script>
          function predict() {
            var request = new XMLHttpRequest();
            request.open("POST", "predict");          
            request.onreadystatechange = function() {
              if(this.readyState === 4 && this.status === 200) {
                var res = JSON.parse(this.responseText);
                if (res.status != 'OK') {
                  document.getElementById("result").innerHTML = res.msg;
                  return;
                }
                var output = '';
                """ + output_postprocess + """
                document.getElementById("result").innerHTML = this.responseText;
                document.getElementById("output").innerHTML = output;
              }
            };          
            var form = document.getElementById("form");
            var formData = new FormData(form);
            request.send(formData);
          }
        </script>
      </head>
      <body>

        <div style="border-radius: 20px; border: 2px solid #888888; padding: 20px;" >
          <div><b>MODEL SIGNATURE : </b>""" + model_name + """</div>
        </div>

        <div style="margin-top: 5px; border-radius: 20px; border: 2px solid #888888; padding: 20px;" >
          <div><b>INPUT(s)</b></div>          
          <form id="form" style="margin-top: 10px;" >
            <div>
              <span>Input Text: </span>
              <span><input type="text" name="text" ></span>
            </div>
            <div style="margin-top: 10px;" >
              <span><button type="button" onclick="predict()">Predict</button></span>
            </div>
          </form>
        </div>

        <div style="margin-top: 5px; border-radius: 20px; border: 2px solid #888888; padding: 20px;" >
          <div><b>API ENDPOINT(s)</b></div>
          <div style="margin-top: 10px;" >""" + example_api_endpoint + """</div>
        </div>

        <div style="margin-top: 5px; border-radius: 20px; border: 2px solid #888888; padding: 20px;" >
          <div><b>OUTPUT(s)</b></div>
          <div id="output" style="margin-top: 10px;" >&nbsp</div>
        </div>

        <div style="margin-top: 5px; border-radius: 20px; border: 2px solid #888888; padding: 20px;" >
          <div><b>RAW OUTPUT(s)</b></div>
          <div id="result" style="margin-top: 10px;" >&nbsp;</div>
        </div>

      </body>
    </html>
  """

class ModelServer():

  def __init__(self, trainer, model, graph, session, model_name):
    self.trainer = trainer
    self.model = model
    self.graph = graph
    self.session = session
    self.model_name = model_name
    
  def start_server(self):
    global server
    server = self
    print('server = ' + str(server))
    app.run(host='0.0.0.0', port=5555)


## unit test ## 
# s = StreamingServer()
# s.start_server(None)

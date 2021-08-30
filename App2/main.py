import os
from googleapiclient import discovery
from flask import Flask, render_template, request
from google.api_core.client_options import ClientOptions
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env', verbose=True)

app = Flask(__name__)

def make_prediction(inputs):


    api_endpoint = 'https://us-east1-ml.googleapis.com'
    bln_local=False
    if bln_local:
        client_options = ClientOptions(api_endpoint=api_endpoint)
        credentials = Credentials.from_service_account_file(os.getenv('GCP_CREDENTIALS_FILE'))
        ml = discovery.build('ml', 'v1', client_options=client_options, credentials=credentials)
    else:
        client_options = ClientOptions(api_endpoint=api_endpoint)
        ml = discovery.build('ml', 'v1', client_options=client_options)

    project_name = 'firstapp-323113'
    model_name = 'regression_insurance_model'
    version_name = 'version1'

    request_body = {'instances': [inputs]}
    prediction_request = ml.projects().predict(
        name='projects/{}/models/{}/versions/{}'.format(project_name, model_name, version_name),
        body=request_body)

    response = prediction_request.execute()
    return response['predictions'][0]


@app.route('/output')
def output_page():
    name = request.args.get("entered_name1")
    inputs = [float(x) for x in name.split(',')]
    prediction = make_prediction(inputs)

    return render_template("output.html", prediction=prediction)



@app.route('/')
def root():

    return render_template('index.html')


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)


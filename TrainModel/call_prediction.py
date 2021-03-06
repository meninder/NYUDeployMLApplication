from google.api_core.client_options import ClientOptions
from google.oauth2.service_account import Credentials
from googleapiclient import discovery
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env', verbose=True)

credentials = Credentials.from_service_account_file('GCP_CREDENTIALS_FILE')
api_endpoint = 'https://us-east1-ml.googleapis.com'

client_options = ClientOptions(api_endpoint=api_endpoint)
ml = discovery.build('ml', 'v1', client_options=client_options, credentials=credentials)

project_name = 'firstapp-323113'
model_name = 'regression_insurance_model'
version_name = 'version1'

request_body = { 'instances': [
    [60, 35],
    [19, 24]
    ]
}
request = ml.projects().predict(
    name='projects/{}/models/{}/versions/{}'.format(project_name, model_name, version_name),
    body=request_body)

response = request.execute()
print(response)








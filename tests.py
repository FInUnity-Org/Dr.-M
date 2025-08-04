import boto3
import json
import numpy as np

# Create a SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Prepare your input data
# The format depends on what your model expects
# Example for a text classification model:
payload = {"text": "This is a sample text for inference"}

# Convert the payload to JSON string
payload_json = json.dumps(payload)

# Invoke the endpoint
response = sagemaker_runtime.invoke_endpoint(
    EndpointName='drmend',
    ContentType='application/json',  # Adjust based on what your model expects
    Body=payload_json
)

# Parse the response
result = json.loads(response['Body'].read().decode())
print(result)
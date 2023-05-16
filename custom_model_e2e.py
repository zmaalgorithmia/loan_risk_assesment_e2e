# %%
import datarobot as dr
from datetime import datetime
import json
import os
import requests
import time

# %% [markdown]
# # Preparation

# %%
MANIFEST_JSON_FILE_PATH = "config/manifest.json"

# %%
# Read metadata of the custom model from the model manifest.json file.
f = open(MANIFEST_JSON_FILE_PATH)

# Returns JSON object as a dictionary
manifest = json.load(f)

print(json.dumps(manifest, indent=4, sort_keys=True, separators=(',', ': ')))

# %%
DATAROBOT_API_ENDPOINT = manifest['datarobot_endpoint']
DATAROBOT_API_KEY = os.getenv("DATAROBOT_API_KEY")

# %%
client = dr.Client(
    token=DATAROBOT_API_KEY,
    endpoint=DATAROBOT_API_ENDPOINT,
)

# %% [markdown]
# # Create Custom Model on DataRobot

# %%
custom_model_manifest = manifest['custom_model']

# %%
# Create a custom model using the metadata specified in the manifest.

if custom_model_manifest['custom_model_id'] == "":
    custom_model = dr.CustomInferenceModel.create(
      name=custom_model_manifest['name'] + " " + datetime.today().strftime('%Y-%m-%d_%H:%M:%S'),
      target_type=custom_model_manifest['target_type'],
      target_name=custom_model_manifest['target_name'],
      language=custom_model_manifest['language']
    )
    
    custom_model_id = custom_model.id
    custom_model_manifest['custom_model_id'] = custom_model_id
    print(f"Custom model ID: {custom_model_id}")

# %%
print(json.dumps(manifest['custom_model'], indent=4, sort_keys=True, separators=(',', ': ')))

# %%
# Specify model environment using model environment id in the manifest.

model_environment_id = manifest['model_environment']['model_environment_id']

if model_environment_id == "":
    raise ValueError("Please provide a model environment id.")
    print("Environment id is missing in the manifest json file.")
    
print(f"Model environment ID: {model_environment_id}")

# %% [markdown]
# ## Create a new version of the custom model

# %%
# Create a new version to hold model content that we are going to upload from the folder path. 
'''
    When running this script from a git repot, for example,  using Jenkins or Bitbucket Pipeline, 
    the folder path points to the git folder where model content for the custom model are stored.
'''

custom_model_version = dr.CustomModelVersion.create_clean(
    custom_model_id = custom_model_id,
    base_environment_id = model_environment_id,
    is_major_update = manifest['custom_model_version']['is_major_update'],
    folder_path= manifest['custom_model_version']['folder_path'], 
)

# %%
custom_model_version.update(
    description = manifest['custom_model_version']['description']
)

# %%
# Capture custom model version ID and add it to the manifest.

custom_model_version_id = custom_model_version.id
manifest['custom_model_version']['version_id'] = custom_model_version_id
print(f"Custom model version ID: {custom_model_version_id}")

# %%
custom_model_version = custom_model_version.label
manifest['custom_model_version']['version'] = custom_model_version
print(f"Custom model version {custom_model_version}")

# %% [markdown]
# ## Build the model environment and dependencies

# %%
# Build model environment with the model content and dependencies for later use. 

build_info = dr.CustomModelVersionDependencyBuild.start_build(
  custom_model_id=custom_model_id,
  custom_model_version_id=custom_model_version_id,
  max_wait=3600,
)

# %%
log = build_info.get_log()

log_file_path = f"{manifest['log_path']}model_build_log_{custom_model_id}_{model_environment_id}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt"

with open(log_file_path, "w") as text_file:
    text_file.write(log)
    
manifest['model_environment']['log_path'] = log_file_path

# %% [markdown]
# ## Associte training data with the custom model

# %%
# Associate training data with the custom model.

if manifest['training_data']['training_data_id'] == "":
    training_data = dr.Dataset.create_from_file(
        file_path = manifest['training_data']['file_path']
    )
    
    training_data_id = training_data.id
    manifest['training_data']['training_data_id'] = training_data_id
else: 
    training_data_id = manifest['training_data']['training_data_id']

print(f"Training dataset id: {training_data_id}")

# %%
custom_model.assign_training_data(training_data_id, partition_column="Partition", max_wait=3600)

# %% [markdown]
# # Run Custom Model Test

# %%
# Provide data for custom model test.

if manifest['model_test']['test_data_id'] == "":
    test_dataset = dr.Dataset.create_from_file(
        file_path = manifest['test_data']['file_path'],
        max_wait = 3600
    )

    test_data_id = test_dataset.id
    manifest['model_test']['test_data_id'] = test_data_id
else:
    test_data_id = manifest['model_test']['test_data_id']

print(f"Test dataset ID: {test_data_id}")

# %%
import time

attempts = 5

while attempts > 0: 
    try:
        custom_model_test = dr.CustomModelTest.create(
          custom_model_id=custom_model_id,
          custom_model_version_id=custom_model_version_id,
          dataset_id=test_data_id,
          max_wait=3600,
        )
    except dr.errors.ClientError as e:
        print(e)
        if "Only 1 active Custom Model Testing is allowed" in str(e):
            attempts -= 1
            print(f"Waiting for 60 seconds to try again...remaining attempts {attemps}")
            time.sleep (60)
    except Exception as e:
        print(e)
    else:
        attempts = 0

print(f"The custom model test is {custom_model_test.overall_status}")

# %%
# Download the log file to a directory. 

log = custom_model_test.get_log()

log_file_path = f"{manifest['log_path']}model_test_log_{custom_model_id}_{custom_model_version}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt"

with open(log_file_path, "w") as text_file:
    text_file.write(log)
    
manifest['model_test']['log_path'] = log_file_path

# %% [markdown]
# # Register a Custom Model Version to Model Registry

# %%
# Register a specific version of the custom mdoel to the model registry.

register_model_package_endpoint = "modelPackages/fromCustomModelVersion/"

request_body = {
    "customModelVersionId": custom_model_version_id,
}

model_registry_resp = client.post(register_model_package_endpoint, data=request_body)

# %%
# Update manifest with the model pakcage id of the registerd custom model.

model_package_id = model_registry_resp.json()['id']

manifest['model_package']['model_package_id'] = model_package_id

print(f"Model Package ID: {model_package_id}")

# %% [markdown]
# # Generate Model Documentation for Custom Model

# %%
# doc = dr.AutomatedDocument(
#     document_type = manifest['model_documentation']['document_type'],
#     entity_id = model_package_id,
#     output_format = manifest['model_documentation']['output_format'],
#     template_id = manifest['model_documentation']['template_id']
# )

# %%
# Automated compliance documentation needs to be initialized first, 
# so that feature assoication, partial dependencies,and such can be calculated by DataRobot.

# if not doc.is_model_compliance_initialized[0]:
#     try:
#         doc.initialize_model_compliance()
#         print("Model documentation has been initialized.")
#     except AsyncTimeoutError as e:
#         while not doc.is_model_compliance_initialized[0]:
#             time.sleep(60)
#         print("Model documentation has been initialized.")
# else:
#     print("Model documentation has been initialized.")

# %%
# model_documentation_request_body = {
#     "entityId": model_package_id,
#     "documentType": manifest['model_documentation']['document_type'],
#     "templateId": manifest['model_documentation']['template_id'],
#     "outputFormat":manifest['model_documentation']['output_format'],
# }

# %%
# MODEL_DOCUMENTATION_ENDPOINT = "automatedDocuments/"

# model_documentation_resp = client.post(MODEL_DOCUMENTATION_ENDPOINT, model_documentation_request_body)

# model_documentation_location = model_documentation_resp.headers['Location']

# print(f"Model documentation location: {model_documentation_location}")

# %%
last_status = ""

# while True:
#     try:
#          status = json.loads(client.get(model_documentation_location).text)['status']
#     except:
#         print("Compliance documentation is generated.")
#         break

#     if status == "INITIALIZED":
#         if last_status == "INITIALIZED":
#             pass
#         else:
#             print("Compliance documentation is being initialized...")
#             last_status = status
#     elif status == "RUNNING":
#         if last_status == "RUNNING":
#             pass
#         else:
#             print("Compliance documentation is being generated...")
#             last_status = status
#     else:
#         print("Compliance documentation is generated.")
#         break

#     time.sleep(10)

# %%
# file_path = f"{manifest['model_documentation']['file_path']}/Model_Documentation_{model_package_id}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.docx"

# print(f"The model documentaiton is stored at {file_path}")

# %%
# generation_response = client.get(model_documentation_location, stream=True)
# with open(file_path, mode="wb") as f:
#     for chunk in generation_response.iter_content(chunk_size=1024 * 1024):
#         f.write(chunk)

# %%
# doc.filepath = file_path
# doc.generate()
# doc.download()

# %% [markdown]
# # Deploy the Custom Model

# %%
CREATE_DEPLOYMENT_FROM_MODEL_PACKAGE_ENDPOINT = "deployments/fromModelPackage/"
DEPLOYMENT_JSON_FILE_PATH = "config/deployment.json"
DEPLOYMENT_UPDATES_JSON_FILE_PATH = "config/deployment_settings.json"

# %%
# Read metadata of the custom model from the model manifest.json file.
f = open(DEPLOYMENT_JSON_FILE_PATH)

# Returns JSON object as a dictionary
deployment_json = json.load(f)

# %%
deployment_json['modelPackageId'] = model_package_id
deployment_json

# %%
def get_id_from_json_response(resp):
    return json.loads(resp.content.decode())['id']

# %%
# Create the deployment
deployment_resp = client.post(CREATE_DEPLOYMENT_FROM_MODEL_PACKAGE_ENDPOINT, data=deployment_json)

# %%
deployment_id = get_id_from_json_response(deployment_resp)

print(f"The deployment id: {deployment_id}")

# %%
def get_setting_from_json(file_path):
    # Read metadata of the model package from the model_package.json file.
    f = open(file_path)

    # Returns JSON object as a dictionary
    settings = json.load(f)

    return settings

# %%
# Read additional deployment settings for drift and accuracy monitoring
deployment_settings_update = get_setting_from_json(DEPLOYMENT_UPDATES_JSON_FILE_PATH)
deployment_settings_update

# %%
# Update the deployment settings for drift and accuracy monitoring
UPDATE_DEPLOYMENT_SETTING_ENDPOINT = f"deployments/{deployment_id}/settings/"
resp = client.patch(UPDATE_DEPLOYMENT_SETTING_ENDPOINT, data=deployment_settings_update)
resp

# %% [markdown]
# # Update Deployment Manifest

# %%
manifest.update({"deployment_settings": deployment_json})

# %%
manifest.update({"deployment_settings_update": deployment_settings_update})

# %%
manifest

# %%
# Write the manifest captured to a json file. 

manifest_file_path = f"log/manifest_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json"
with open(manifest_file_path, 'w') as mf:
    json.dump(manifest, mf, indent=4)
    
print(f"The updated manifest is stored at {manifest_file_path}")

# %% [markdown]
# # Clean up

# %%
clean_up = True

# %%
if clean_up:
    dr.Deployment.get(deployment_id).delete()
    
    ARCHIVE_MODEL_PACKAGE_ENDPOINT = f"modelPackages/{model_package_id}/archive/"
    client.post(ARCHIVE_MODEL_PACKAGE_ENDPOINT)
    
    custom_model.delete()

# %%




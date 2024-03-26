# Databricks notebook source
# MAGIC %md
# MAGIC # Deploying our Chat Model to the model serving endpoint
# MAGIC

# COMMAND ----------

# MAGIC %pip install databricks-sdk mlflow gradio requests
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init-advanced $reset_all_data=false

# COMMAND ----------

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
bot_name = dbutils.widgets.get("bot_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the model to model serving endpoint

# COMMAND ----------

# DBTITLE 1,Takes around 20 mins
import urllib
import json
import mlflow

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

mlflow.set_registry_uri('databricks-uc')
client = MlflowClient()
model_name = f"{catalog_name}.{schema_name}.{bot_name}"
serving_endpoint_name = f"dw_demo_{bot_name}"[:63]
latest_model = client.get_model_version_by_alias(model_name, "champion") #manually change the mode with the alias champion

w = WorkspaceClient()
#TODO: use the sdk once model serving is available.
serving_client = EndpointApiClient()
# Start the endpoint using the REST API (you can do it using the UI directly)
auto_capture_config = {
    "catalog_name": catalog_name,
    "schema_name": schema_name,
    "table_name_prefix": serving_endpoint_name
    }
environment_vars={"DATABRICKS_TOKEN": "{{secrets/dbdemos/dw-rag-demo}}"} # needs to be in secret, my own PAT
serving_client.create_endpoint_if_not_exists(serving_endpoint_name, model_name=model_name, model_version = latest_model.version, workload_size="Small", scale_to_zero_enabled=True, wait_start = True, auto_capture_config=auto_capture_config, environment_vars=environment_vars)

# COMMAND ----------

displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# DBTITLE 1,Inference test
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import DataframeSplitInput

df_split = DataframeSplitInput(columns=["messages"],
                               data=[[ {"messages": [{"role": "user", "content": "What is the ICD-10 code of        unconscious?"}, 
                                                     {"role": "assistant", "content": "The ICD-10 code for 'Unspecified coma' is R40.20. This code is used when a person is in a coma, but the specific type or cause of the coma is not known. It falls under the category of symptoms and signs involving cognition, perception, emotional state, and behavior. The code R40.20 is applicable to situations where a patient presents with unconsciousness or coma without further specification, making it a crucial code for medical documentation, clinical diagnosis, and billing purposes."}, 
                                                     {"role": "user", "content": "What is the ICD-10 code of skull fracture?"}
                                                    ]}]])
w = WorkspaceClient()
w.serving_endpoints.query(serving_endpoint_name, dataframe_split=df_split)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gradio front end UI
# MAGIC * If you just want to run Gradio, run cell 1-4 then skip to this session

# COMMAND ----------

import gradio as gr
import requests
bot_name = dbutils.widgets.get("bot_name")
serving_endpoint_name = f"dw_demo_{bot_name}"[:63]

# COMMAND ----------

def make_prediction(input_text):
  from databricks.sdk import WorkspaceClient
  from databricks.sdk.service.serving import DataframeSplitInput

  df_split = DataframeSplitInput(columns=["messages"],
                                data=[[ {"messages": [{"role": "user", "content": "What is the ICD-10 code of        unconscious?"}, 
                                                      {"role": "assistant", "content": "The ICD-10 code for 'Unspecified coma' is R40.20. This code is used when a person is in a coma, but the specific type or cause of the coma is not known. It falls under the category of symptoms and signs involving cognition, perception, emotional state, and behavior. The code R40.20 is applicable to situations where a patient presents with unconsciousness or coma without further specification, making it a crucial code for medical documentation, clinical diagnosis, and billing purposes."}, 
                                                      {"role": "user", "content": f"{input_text}"}
                                                      ]}]])
  w = WorkspaceClient()
  predicted_output = w.serving_endpoints.query(serving_endpoint_name, dataframe_split=df_split)
  return predicted_output.as_dict()['predictions'][0]['result']

# COMMAND ----------

# DBTITLE 1,Test the function
make_prediction("What is the ICD-10 code for fracture skull and the reasoning?")

# COMMAND ----------

# DBTITLE 1,Gradio UI
iface = gr.Interface(
    fn=make_prediction,
    inputs=gr.Textbox(label='Question'),
    outputs=gr.Textbox(label='Response'),
    title="Databricks Serving Endpoint Demo",
    description="Enter your question and click Submit to make a prediction",
    theme="freddyaboulton/dracula_revamped" #huggingface template
)

# Launch the Gradio interface
iface.launch(share=True, debug=True) # for some reason need to run in debug mode to work

# COMMAND ----------



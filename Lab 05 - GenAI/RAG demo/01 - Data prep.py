# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Ingesting and preparing PDF for LLM and Self Managed Vector Search Embeddings
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-0.png?raw=true" style="float: right; width: 600px; margin-left: 10px">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment Setup

# COMMAND ----------

# DBTITLE 1,Install libraries
# MAGIC %pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.0.319 llama-index==0.9.3 databricks-vectorsearch==0.20 pydantic==1.10.9 mlflow==2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,import functions
# MAGIC %run ./_resources/00-init-advanced $reset_all_data=false

# COMMAND ----------

# DBTITLE 1,Create catalog, schema and volume
# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS ${catalog_name};
# MAGIC CREATE SCHEMA IF NOT EXISTS ${catalog_name}.${schema_name};
# MAGIC CREATE VOLUME IF NOT EXISTS ${catalog_name}.${schema_name}.${volume_name};

# COMMAND ----------

# MAGIC %md
# MAGIC ## PDF ingestions

# COMMAND ----------

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
volume_name = dbutils.widgets.get("volume_name")
bronze_table = dbutils.widgets.get("bronze_table")
silver_table = dbutils.widgets.get("silver_table")
vector_search = dbutils.widgets.get("vector_search")
index_name = dbutils.widgets.get("index_name")

# COMMAND ----------

# DBTITLE 1,Check the content in the volume

volume_folder =  f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"
display(dbutils.fs.ls(f"{volume_folder}"))

# COMMAND ----------

# DBTITLE 1,Drop existing raw table (pdf text extract)
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS ${catalog_name}.${schema_name}.${bronze_table};

# COMMAND ----------

dbutils.fs.rm(f"{volume_folder}/checkpoints/{volume_name}-checkpoint", True)
dbutils.fs.rm(f"{volume_folder}/checkpoints/{volume_name}-checkpoint-2", True)

# COMMAND ----------

df = (spark.readStream
        .format('cloudFiles')
        .option('cloudFiles.format', 'BINARYFILE')
        .option("pathGlobFilter", "*.pdf")
        .load('dbfs:'+volume_folder))

# Write the data as a Delta table
(df.writeStream
  .trigger(availableNow=True)
  .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/{volume_name}-checkpoint')
  .table(f"{catalog_name}.{schema_name}.{bronze_table}").awaitTermination())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ${catalog_name}.${schema_name}.${bronze_table};

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract PDF context as text chunks

# COMMAND ----------

# For production use-case, install the libraries at your cluster level with an init script instead. 
install_ocr_on_nodes()

# COMMAND ----------

# DBTITLE 1,Transform pdf as text
from unstructured.partition.auto import partition
import re

def extract_doc_text(x : bytes) -> str:
  # Read files and extract the values with unstructured
  sections = partition(file=io.BytesIO(x))
  def clean_section(txt):
    txt = re.sub(r'\n', '', txt)
    return re.sub(r' ?\.', '.', txt)
  # Default split is by section of document, concatenate them all together because we want to split by sentence instead.
  return "\n".join([clean_section(s.text) for s in sections]) 

# COMMAND ----------

import io
import re
from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from llama_index import Document, set_global_tokenizer
from transformers import AutoTokenizer

# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #set llama2 as tokenizer to match our model size (will stay below BGE 1024 limit)
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    #Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    def extract_and_split(b):
      txt = extract_doc_text(b)
      nodes = splitter.get_nodes_from_documents([Document(text=txt)])
      return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks BGE Embeddings Foundation Model Endpoint

# COMMAND ----------

# DBTITLE 1,Using Databricks Foundation model BGE as embedding endpoint
from mlflow.deployments import get_deploy_client

# bge-large-en Foundation models are available using the /serving-endpoints/databricks-bge-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

## NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What are the steps a practitioner should follow when reporting a suspected case of child abuse or neglect?"]})
pprint(embeddings)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS ${catalog_name}.${schema_name}.${silver_table};

# COMMAND ----------

# DBTITLE 1,Create the final databricks_pdf_documentation table containing chunks and embeddings
# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS ${catalog_name}.${schema_name}.${silver_table} (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Computing the chunk embeddings and saving them to our Delta Table

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

(spark.readStream.table(f"{catalog_name}.{schema_name}.{bronze_table}")
      .withColumn("content", F.explode(read_as_chunk("content")))
      .withColumn("embedding", get_embedding("content"))
      .selectExpr('path as url', 'content', 'embedding')
  .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/{volume_name}-checkpoint-2')
    .table(f'{catalog_name}.{schema_name}.{silver_table}').awaitTermination())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ${catalog_name}.${schema_name}.${silver_table} WHERE url like '%.pdf' limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Search

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME=f"{vector_search}"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

# COMMAND ----------

# DBTITLE 1,Create the vector search endpoint
# vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

# wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
# print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ${catalog_name}.${schema_name}.${silver_table}

# COMMAND ----------

# DBTITLE 1,Create the Self-managed vector search using our endpoint
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = f"{catalog_name}.{schema_name}.{silver_table}"
# Where we want to store our index
vs_index_fullname = f"{catalog_name}.{schema_name}.{index_name}"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED", #Sync needs to be manually triggered
    primary_key="id",
    embedding_dimension=1024, #Match your model embedding size (bge)
    embedding_vector_column="embedding"
  )
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

#Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Searching for similar content

# COMMAND ----------

# DBTITLE 1,Similarity search
from mlflow.deployments import get_deploy_client
deploy_client = get_deploy_client("databricks")

question = "What are the steps a practitioner should follow when reporting a suspected case of child abuse or neglect?"

response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["url", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
pprint(docs)

# COMMAND ----------



# Databricks notebook source
# MAGIC %md # Building your first RAG
# MAGIC
# MAGIC This is a notebook to create your first RAG Application

# COMMAND ----------

# MAGIC %pip install -U langchain==0.1.10 sqlalchemy==2.0.27 pypdf==4.1.0 mlflow databricks-vectorsearch 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Params 
db_catalog = 'gen_ai_catalog'
db_schema = 'lab_05'
db_table = "arxiv_parse"

endpoint_name = "workshop-vs-endpoint"
vs_index = f"{db_table}_bge_index"
vs_index_fullname = f"{db_catalog}.{db_schema}.{vs_index}"

# temp need to change later
embedding_model = "databricks-bge-large-en"
chat_model = "databricks-mixtral-8x7b-instruct"

# COMMAND ----------

# Construct and Test LLM Chain
from langchain_community.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import DatabricksEmbeddings

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# this is for DB FM API only not Provisioned Throughput
# FM API is US only for now
chat = ChatDatabricks(
    target_uri="databricks",
    endpoint=chat_model,
    temperature=0.1,
)

# Test that it is working
chat([HumanMessage(content="hello")])

# COMMAND ----------

# Construct and Test Embedding Model
embeddings = DatabricksEmbeddings(endpoint=embedding_model)
embeddings.embed_query("hello")[:3]

# COMMAND ----------

# MAGIC %md ## Setting Up chain

# COMMAND ----------

vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=endpoint_name,
                      index_name=vs_index_fullname)

retriever = DatabricksVectorSearch(
    index, text_column="page_content", 
    embedding=embeddings, columns=["source_doc"]
).as_retriever()

retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a response to answer this question")
    ])

retriever_chain = create_history_aware_retriever(chat, retriever, retriever_prompt)

# COMMAND ----------

prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
         You are a research and eduation assistant that takes a conversation between a learner and yourself and answer their questions based on the below context:
         
         <context>
         {context}
         </context> 
         
         <instructions>
         - Focus your answers based on the context but provide additional helpful notes from your background knowledge caveat those notes though.
         - If the context does not seem relevant to the answer say as such as well.
         </instructions>
         """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

stuff_documents_chain = create_stuff_documents_chain(chat, prompt_template)

retrieval_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)

# COMMAND ----------

stuff_documents_chain

# COMMAND ----------

retrieval_chain

# COMMAND ----------

# Testing stuff chain
retrieval_chain.invoke(
  {
    "chat_history": [], 
    "input": "Tell me about tuning LLMs", 
    "context": "",
  }
)["answer"]

# COMMAND ----------

# MAGIC %md ## MLFlow Logging
# MAGIC

# COMMAND ----------

def model(input_df):
    answer = []
    for index, row in input_df.iterrows():
        answer.append(
            retrieval_chain.invoke(
                {
                    "chat_history": [], 
                    "input": row["questions"], 
                    "context": ""
                }
            )["answer"]
        )
    return answer

# COMMAND ----------

import pandas as pd
import mlflow

# COMMAND ----------

eval_df = pd.DataFrame(
    {
        "questions": [
            "What is a LLM?",
            "How to run mlflow.evaluate()?",
            "How to log_table()?",
            "How to load_table()?",
        ],
        "context": [
            "",
            "",
            "",
            "",
        ],
    }
)

# COMMAND ----------

from mlflow.metrics.genai import EvaluationExample, faithfulness

# Create a good and bad example for faithfulness in the context of this problem
faithfulness_examples = [
    EvaluationExample(
        input="What is a LLM?",
        output="mlflow.autolog(disable=True) will disable autologging for all functions. In Databricks, autologging is enabled by default. ",
        score=2,
        justification="The output provides a working solution, using the mlflow.autolog() function that is provided in the context.",
        grading_context={
            "context": "mlflow.autolog(log_input_examples: bool = False, log_model_signatures: bool = True, log_models: bool = True, log_datasets: bool = True, disable: bool = False, exclusive: bool = False, disable_for_unsupported_versions: bool = False, silent: bool = False, extra_tags: Optional[Dict[str, str]] = None) → None[source] Enables (or disables) and configures autologging for all supported integrations. The parameters are passed to any autologging integrations that support them. See the tracking docs for a list of supported autologging integrations. Note that framework-specific configurations set at any point will take precedence over any configurations set by this function."
        },
    ),
    EvaluationExample(
        input="How do I disable MLflow autologging?",
        output="mlflow.autolog(disable=True) will disable autologging for all functions.",
        score=5,
        justification="The output provides a solution that is using the mlflow.autolog() function that is provided in the context.",
        grading_context={
            "context": "mlflow.autolog(log_input_examples: bool = False, log_model_signatures: bool = True, log_models: bool = True, log_datasets: bool = True, disable: bool = False, exclusive: bool = False, disable_for_unsupported_versions: bool = False, silent: bool = False, extra_tags: Optional[Dict[str, str]] = None) → None[source] Enables (or disables) and configures autologging for all supported integrations. The parameters are passed to any autologging integrations that support them. See the tracking docs for a list of supported autologging integrations. Note that framework-specific configurations set at any point will take precedence over any configurations set by this function."
        },
    ),
]

faithfulness_metric = faithfulness(
    model="endpoints:/databricks-llama-2-70b-chat", examples=faithfulness_examples
)
print(faithfulness_metric)


# COMMAND ----------

from mlflow.metrics.genai import EvaluationExample, relevance

relevance_metric = relevance(model="endpoints:/databricks-llama-2-70b-chat")
print(relevance_metric)


# COMMAND ----------

results = mlflow.evaluate(
    model,
    eval_df,
    model_type="question-answering",
    evaluators="default",
    predictions="result",
    extra_metrics=[faithfulness_metric, relevance_metric, mlflow.metrics.latency()],
    evaluator_config={
        "col_mapping": {
            "inputs": "questions",
            "context": "context",
        }
    },
)
print(results.metrics)


# COMMAND ----------

results.tables["eval_results_table"]


# COMMAND ----------

results.tables["eval_results_table"]["outputs"][0]

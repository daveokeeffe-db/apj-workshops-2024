# Databricks notebook source
# MAGIC %md # Building your first RAG
# MAGIC
# MAGIC This is a notebook to create your first RAG Application

# COMMAND ----------

# MAGIC %pip install -U langchain==0.1.13 sqlalchemy==2.0.27 pypdf==4.1.0 mlflow==2.10 databricks-vectorsearch gradio requests
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Params 
db_catalog = 'gen_ai_catalog'
db_schema = 'lab_05'
db_table = "arxiv_parse"

endpoint_name = "one-env-shared-endpoint-1"
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
chat([HumanMessage(content="What is a LLM?")])

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
         - Focus your answers based on the context but provide additional helpful notes from your background knowledge.
         </instructions>
         """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

stuff_documents_chain = create_stuff_documents_chain(chat, prompt_template)

retrieval_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)

# COMMAND ----------

# Testing stuff chain
def rag_chat(input_text, chat_history=[], context=""):
  return retrieval_chain.invoke(
    {
      "chat_history": chat_history, 
      "input": input_text, 
      "context": context,
    }
  )["answer"]

# COMMAND ----------

rag_chat("What is a LLM?", context="An LLM is a Large Language Model")

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
                    "context": row["context"],
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
            "Tell me about tuning LLMs",
            "What is Flava?",
            "Why should I compress an LLM?",
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

from mlflow.metrics.genai import EvaluationExample

professionalism_example_score_2 = mlflow.metrics.genai.EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps "
        "you track experiments, package your code and models, and collaborate with your team, making the whole ML "
        "workflow smoother. It's like your Swiss Army knife for machine learning!"
    ),
    score=2,
    justification=(
        "The response is written in a casual tone. It uses contractions, filler words such as 'like', and "
        "exclamation points, which make it sound less professional. "
    ),
)
professionalism_example_score_4 = mlflow.metrics.genai.EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It was "
        "developed by Databricks, a company that specializes in big data and machine learning solutions. MLflow is "
        "designed to address the challenges that data scientists and machine learning engineers face when "
        "developing, training, and deploying machine learning models.",
    ),
    score=4,
    justification=("The response is written in a formal language and a neutral tone. "),
)

professionalism = mlflow.metrics.genai.make_genai_metric(
    name="healthiness",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is"
        "tailored to the context and audience. It often involves avoiding overly casual language, slang, or "
        "colloquialisms, and instead using clear, concise, and respectful language."
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below are the details for different scores: "
        "- Score 0: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for "
        "professional contexts."
        "- Score 1: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
        "some informal professional settings."
        "- Score 2: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
        "- Score 3: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 4: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal "
        "business or academic settings. "
    ),
    examples=[professionalism_example_score_2, professionalism_example_score_4],
    model="endpoints:/azure-openai-gpt4",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    greater_is_better=True,
)

# COMMAND ----------

from mlflow.metrics.genai import EvaluationExample, relevance

relevance_metric = relevance(model="endpoints:/azure-openai-gpt4")
endpoint_type = "llm/v1/completions"

# COMMAND ----------

results = mlflow.evaluate(
    model,
    eval_df,
    model_type="question-answering",
    evaluators="default",
    predictions="result",
    extra_metrics=[professionalism, relevance_metric, mlflow.metrics.latency()],
    evaluator_config={
        "col_mapping": {
            "inputs": "questions",
            "context": "context",
        }
    },
)

# COMMAND ----------

results.__dict__

# COMMAND ----------

results.tables["eval_results_table"]


# COMMAND ----------

import mlflow
catalog = "main"
schema = "default"
model_name = "my_model"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model("runs:/ec42300abad44bb6b9856f34edd99e29/model", f"{db_catalog}.{db_schema}.{model_name}")

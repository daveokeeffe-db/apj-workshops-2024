# Databricks notebook source
# MAGIC %md # Building your first RAG
# MAGIC
# MAGIC This is a notebook to create your first RAG Application

# COMMAND ----------

pip install -U langchain==0.1.13 sqlalchemy==2.0.27 pypdf==4.1.0 mlflow databricks-vectorsearch gradio requests
dbutils.library.restartPython()

# COMMAND ----------

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

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
def rag_chat(input_text, context=""):
  return retrieval_chain.invoke(
    {
      "chat_history": [], 
      "input": input_text, 
      "context": context,
    }
  )["answer"]

# COMMAND ----------

rag_chat("What is a LLM?", context="An LLM is a Large Language Model")

# COMMAND ----------

import gradio as gr

# COMMAND ----------

iface = gr.Interface(
    fn=rag_chat,
    inputs=[gr.Textbox(label='Question'), gr.Textbox(label='Context')],
    outputs=gr.Textbox(label='Response'),
    title="Databricks Serving Endpoint Demo",
    description="Enter your question and click Submit to make a prediction",
    theme="freddyaboulton/dracula_revamped" #huggingface template
)

# Launch the Gradio interface
iface.launch(share=True, debug=True) # for some reason need to run in debug mode to work

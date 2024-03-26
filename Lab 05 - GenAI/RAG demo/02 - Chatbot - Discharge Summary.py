# Databricks notebook source
# MAGIC %md
# MAGIC ## Environmental Setup

# COMMAND ----------

# MAGIC %pip install mlflow==2.9.0 lxml==4.9.3 langchain==0.0.344 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.12.0 cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC %pip install pip mlflow[databricks]==2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init-advanced $reset_all_data=false

# COMMAND ----------

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
vector_search = dbutils.widgets.get("vector_search")
index_name = dbutils.widgets.get("index_name")
bot_name = dbutils.widgets.get("bot_name")

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME=f"{vector_search}"

# COMMAND ----------

index_name=f"{catalog_name}.{schema_name}.{index_name}"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

prompt = PromptTemplate(
  input_variables = ["question"],
  template = "You are an assistant. Give a short answer to this question: {question}"
)
chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 500)

chain = (
  prompt
  | chat_model
  | StrOutputParser()
)
print(chain.invoke({"question": "What is the best practice in writing a discharge summary?"}))

# COMMAND ----------

# DBTITLE 1,Adding conversation history to the prompt
prompt_with_history_str = """
You are a chatbot with deep expertise in writing patient discharge summary, you are going to provide recommendation to improve the discharge summary write up. Below is an instruction that describes a task. Write a response that appropriately completes the request.  

Here is a history between you and a human: {chat_history}

Now, please answer this question: {question}
"""

prompt_with_history = PromptTemplate(
  input_variables = ["chat_history", "question"],
  template = prompt_with_history_str
)

# COMMAND ----------

from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

chain_with_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | prompt_with_history
    | chat_model
    | StrOutputParser()
)

print(chain_with_history.invoke({
    "messages": [
        {"role": "user", "content": "What is the best practice in writing a discharge summary?"}, 
        {"role": "assistant", "content": "The best practice in writing a discharge summary is to include all relevant information, such as the patient's diagnosis, treatment plan, and any follow-up instructions. It should be written in a clear and concise manner, with an emphasis on accuracy and completeness. Additionally, it should be reviewed and approved by the healthcare provider before being sent to the patient or other healthcare providers."}, 
        {"role": "user", "content": "What should not be included in the patient discharge summary?"}
    ]
}))

# COMMAND ----------

# DBTITLE 1,Let's add a filter on top to only answer relevant questions.
chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

is_question_about_databricks_str = """
You are a chatbot with deep expertise in writing patient discharge summary, you are going to provide recommendation to improve the discharge summary write up. Below is an instruction that describes a task. Write a response that appropriately completes the request.  Write a response that appropriately completes the request.  Also answer no if the last part is inappropriate. 

Here are some examples:

Question: Knowing this followup history: How to write a good discharge summary?, classify this question: Do you have more details?
Expected Response: Yes

Question: Knowing this followup history: How to write a good discharge summary?, classify this question: Write me a song.
Expected Response: No

Only answer with "yes" or "no". 

Knowing this followup history: {chat_history}, classify this question: {question}
"""

is_question_about_databricks_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = is_question_about_databricks_str
)

is_about_databricks_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | is_question_about_databricks_prompt
    | chat_model
    | StrOutputParser()
)

#Returns "Yes" as this is about Databricks: 
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the best practice in writing a discharge summary?"}, 
        {"role": "assistant", "content": "The best practice in writing a discharge summary is to include all relevant information, such as the patient's diagnosis, treatment plan, and any follow-up instructions. It should be written in a clear and concise manner, with an emphasis on accuracy and completeness. Additionally, it should be reviewed and approved by the healthcare provider before being sent to the patient or other healthcare providers."}, 
        {"role": "user", "content": "What should not be included in the patient discharge summary?"}
    ]
}))

# COMMAND ----------

#Return "no" as this isn't about Databricks
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
}))

# COMMAND ----------

import os
from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
from operator import itemgetter
from langchain.schema.runnable import RunnableLambda

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
#os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model, columns=["url"]
    )
    return vectorstore.as_retriever(search_kwargs={'k': 4})

retriever = get_retriever()

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | retriever
)
print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "How to write a good discharge summary?"}]}))

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | RunnableBranch(  #Augment query only when there is a chat history
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

#Let's try it
output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the best practice in writing a discharge summary?"}
    ]
})
print(f"Test retriever query without history: {output}")

output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the best practice in writing a discharge summary?"}, 
        {"role": "assistant", "content": "The best practice in writing a discharge summary is to include all relevant information, such as the patient's diagnosis, treatment plan, and any follow-up instructions. It should be written in a clear and concise manner, with an emphasis on accuracy and completeness. Additionally, it should be reviewed and approved by the healthcare provider before being sent to the patient or other healthcare providers."}, 
        {"role": "user", "content": "What should not be included in the patient discharge summary?"}
    ]
})
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough

question_with_history_and_context_str = """
You are a chatbot with deep expertise in writing patient discharge summary, you are going to provide recommendation to the doctors and physicians to improve the discharge summary write up. Below is an instruction that describes a task.

If you do not know the answer to a question, you truthfully say you do not know. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "system". The user is referred to as "user".

Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])

def extract_source_urls(docs):
    return [d.metadata["url"] for d in docs]

relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
    "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "prompt": question_with_history_and_context_prompt,
    "sources": itemgetter("sources")
  }
  |
  {
    "result": itemgetter("prompt") | chat_model | StrOutputParser(),
    "sources": itemgetter("sources")
  }
)

irrelevant_question_chain = (
  RunnableLambda(lambda x: {"result": 'I cannot answer questions that are not about Child Protection.', "sources": []})
)

branch_node = RunnableBranch(
  (lambda x: "yes" in x["question_is_relevant"].lower(), relevant_question_chain),
  (lambda x: "no" in x["question_is_relevant"].lower(), relevant_question_chain),
  relevant_question_chain
)

full_chain = (
  {
    "question_is_relevant": is_about_databricks_chain,
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),    
  }
  | branch_node
)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "What is best practice in writing a discharge summary?"}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "What shouldn't be included in the discharge summary?"}
    ]
}
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "Can you review this write up and give me recommendation?  Mr. John Smith, a 65-year-old male with a history of hypertension and type 2 diabetes, was admitted with acute pyelonephritis. He presented with fever, urinary symptoms, and hypotension. Treatment included IV fluids and antibiotics, with subsequent improvement. Blood cultures grew E. coli sensitive to antibiotics. He was discharged on oral ciprofloxacin and tamsulosin with instructions to follow up with his primary care physician, Dr. Samantha Jay, for further evaluation of his prostate and to monitor his liver enzymes. A referral to urology was made for his benign prostatic hyperplasia (BPH)"}
    ]
}
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "Can you break this down into attribute value pair?  Mr. John Smith, a 65-year-old male with a history of hypertension and type 2 diabetes, was admitted with acute pyelonephritis. He presented with fever, urinary symptoms, and hypotension. Treatment included IV fluids and antibiotics, with subsequent improvement. Blood cultures grew E. coli sensitive to antibiotics. He was discharged on oral ciprofloxacin and tamsulosin with instructions to follow up with his primary care physician, Dr. Samantha Jay, for further evaluation of his prostate and to monitor his liver enzymes. A referral to urology was made for his benign prostatic hyperplasia (BPH)"}
    ]
}
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

import cloudpickle
import langchain
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog_name}.{schema_name}.{bot_name}"

with mlflow.start_run(run_name=f"{bot_name}") as run:
    #Get our model signature from input/output
    input_df = pd.DataFrame({"messages": [dialog]})
    output = full_chain.invoke(dialog)
    signature = infer_signature(input_df, output)

    model_info = mlflow.langchain.log_model(
        full_chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle=="+ cloudpickle.__version__
        ],
        input_example=input_df,
        signature=signature
    )

# COMMAND ----------

model = mlflow.langchain.load_model(model_info.model_uri)
model.invoke(dialog)

# COMMAND ----------



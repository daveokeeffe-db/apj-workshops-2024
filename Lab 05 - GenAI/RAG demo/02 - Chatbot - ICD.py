# Databricks notebook source
# MAGIC %md
# MAGIC # Advanced chatbot with message history and filter using Langchain
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-flow-2.png?raw=true" style="float: right; margin-left: 10px"  width="900px;">

# COMMAND ----------

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
print(chain.invoke({"question": "What is the ICD-10 code of unconscious?"}))

# COMMAND ----------

# DBTITLE 1,Adding conversation history to the prompt
prompt_with_history_str = """
You are a chatbot with deep expertise in ICD-10 code, you are going to provide recommended ICD-10 codes based on the symptoms and findings. Below is an instruction that describes a task. Write a response that appropriately completes the request.  

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
        {"role": "user", "content": "What is the ICD-10 code of unconscious?"}, 
        {"role": "assistant", "content": "The ICD-10 code for 'Unspecified coma' is R40.20. This code is used when a person is in a coma, but the specific type or cause of the coma is not known. It falls under the category of symptoms and signs involving cognition, perception, emotional state, and behavior. The code R40.20 is applicable to situations where a patient presents with unconsciousness or coma without further specification, making it a crucial code for medical documentation, clinical diagnosis, and billing purposes."}, 
        {"role": "user", "content": "What is the ICD-10 code of skull fracture?"}
    ]
}))

# COMMAND ----------

# DBTITLE 1,Let's add a filter on top to only answer relevant questions.
chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

is_question_about_databricks_str = """
You are a chatbot with deep expertise in ICD-10 code, you are going to provide recommended ICD-10 codes based on the symptoms and findings. Below is an instruction that describes a task. Write a response that appropriately completes the request.  Write a response that appropriately completes the request.  Also answer no if the last part is inappropriate. 

Here are some examples:

Question: Knowing this followup history: What is the ICD-10 code of unconscious?, classify this question: Do you have more details?
Expected Response: Yes

Question: Knowing this followup history: What is the ICD-10 code of unconscious?, classify this question: Write me a song.
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
        {"role": "user", "content": "What is the ICD-10 code of unconscious?"}, 
        {"role": "assistant", "content": "The ICD-10 code for 'Unspecified coma' is R40.20. This code is used when a person is in a coma, but the specific type or cause of the coma is not known. It falls under the category of symptoms and signs involving cognition, perception, emotional state, and behavior. The code R40.20 is applicable to situations where a patient presents with unconsciousness or coma without further specification, making it a crucial code for medical documentation, clinical diagnosis, and billing purposes."}, 
        {"role": "user", "content": "What is the ICD-10 code of skull fracture?"}
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
print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What is the ICD-10 code of skull fracture?"}]}))

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
        {"role": "user", "content": "What is the ICD-10 code of unconscious?"}
    ]
})
print(f"Test retriever query without history: {output}")

output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the ICD-10 code of unconscious?"}, 
        {"role": "assistant", "content": "The ICD-10 code for 'Unspecified coma' is R40.20. This code is used when a person is in a coma, but the specific type or cause of the coma is not known. It falls under the category of symptoms and signs involving cognition, perception, emotional state, and behavior. The code R40.20 is applicable to situations where a patient presents with unconsciousness or coma without further specification, making it a crucial code for medical documentation, clinical diagnosis, and billing purposes."}, 
        {"role": "user", "content": "What is the ICD-10 code of skull fracture?"}
    ]
})
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough

question_with_history_and_context_str = """
You are a chatbot with deep expertise in ICD-10 code, you are going to provide recommended ICD-10 codes based on the symptoms and findings.  Below is an instruction that describes a task.

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
        {"role": "user", "content": "What is the ICD-10 code of skull fracture and the reason for this mapping?"}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "A 52-year-old male patient presents to the clinic with a two-week history of coughing up blood, or hemoptysis. He reports producing a teaspoon of bright red blood with each coughing episode. The patient has a history of smoking a pack of cigarettes daily for the past 30 years and has recently experienced unintentional weight loss and night sweats. Upon examination, he appears mildly dyspneic with decreased breath sounds in the right upper lobe of the lung.  Give me the ICD-10 code and the reasoning."}
    ]
}
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "A 25-year-old female patient presents to the emergency department with acute onset of nausea and vomiting that began several hours after consuming sushi at a local restaurant. She reports vomiting multiple times, unable to keep down liquids, and describes her vomit as initially containing food particles, later becoming clear and bile-stained. The patient also complains of mild abdominal cramps but denies any fever, diarrhea, or blood in the vomit. She has no significant past medical history and takes no medications. On examination, she is mildly dehydrated but hemodynamically stable.  What is the ICD-10 for this patient and tell me why?"}
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



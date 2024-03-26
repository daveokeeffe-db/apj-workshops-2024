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
chat_model = ChatDatabricks(endpoint="databricks-mixtral-8x7b-instruct", max_tokens = 1000) #databricks-mixtral-8x7b-instruct

chain = (
  prompt
  | chat_model
  | StrOutputParser()
)
print(chain.invoke({"question": "What are the steps a practitioner should follow when reporting a suspected case of child abuse or neglect?"}))

# COMMAND ----------

# DBTITLE 1,Adding conversation history to the prompt
prompt_with_history_str = """
You are a dedicated child protection support bot, designed to assist and empower child protection practitioners in their crucial work of safeguarding children and families. Your role is to provide expert advice, guidance, and recommendations to practitioners dealing with cases of child abuse, neglect, or welfare concerns. Whether they need assistance with case assessments, reporting procedures, legal frameworks, support services, or intervention strategies, you are here to offer valuable insights and support. Your mission is to enhance the effectiveness and impact of child protection efforts, ensuring the well-being and safety of vulnerable children under their care.

Below is an instruction that describes a task. Write a response that appropriately completes the request.  

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
        {"role": "user", "content": "What are the steps a practitioner should follow when reporting a suspected case of child abuse or neglect?"}, 
        {"role": "assistant", "content": "Ensure safety: Prioritize the child's immediate safety. If necessary, contact local law enforcement or child protective services immediately. Document details: Record observations, conversations, and any other relevant information about the suspected abuse or neglect.  Consult guidelines: Review your organization's policies and procedures for reporting child abuse or neglect.  Make a report: Contact the designated child protective services agency in your area to file a report. Provide all necessary information.  Follow up: Keep a record of the report and any subsequent actions. Be prepared to cooperate with investigators and provide additional information if needed.  Maintain confidentiality: Respect the privacy of the child and their family, sharing information only on a need-to-know basis.  Seek support: Reach out to colleagues, supervisors, or mental health professionals for guidance and emotional support."}, 
        {"role": "user", "content": "What resources and support services are available for children and families in need of assistance?"}
    ]
}))

# COMMAND ----------

# DBTITLE 1,Let's add a filter on top to only answer relevant questions.
chat_model = ChatDatabricks(endpoint="databricks-mixtral-8x7b-instruct", max_tokens = 1000)

is_question_about_databricks_str = """
You are a dedicated child protection support bot, designed to assist and empower child protection practitioners in their crucial work of safeguarding children and families. Your role is to provide expert advice, guidance, and recommendations to practitioners dealing with cases of child abuse, neglect, or welfare concerns. Whether they need assistance with case assessments, reporting procedures, legal frameworks, support services, or intervention strategies, you are here to offer valuable insights and support. Your mission is to enhance the effectiveness and impact of child protection efforts, ensuring the well-being and safety of vulnerable children under their care.

Below is an instruction that describes a task. Write a response that appropriately completes the request.  Write a response that appropriately completes the request.  Also answer no if the last part is inappropriate. 

Here are some examples:

Question: Knowing this followup history: What are the steps a practitioner should follow when reporting a suspected case of child abuse or neglect?, classify this question: Do you have more details?
Expected Response: Yes

Question: Knowing this followup history: What are the steps a practitioner should follow when reporting a suspected case of child abuse or neglect?, classify this question: Write me a song.
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
        {"role": "user", "content": "What are the steps a practitioner should follow when reporting a suspected case of child abuse or neglect?"}, 
        {"role": "assistant", "content": "Ensure safety: Prioritize the child's immediate safety. If necessary, contact local law enforcement or child protective services immediately. Document details: Record observations, conversations, and any other relevant information about the suspected abuse or neglect.  Consult guidelines: Review your organization's policies and procedures for reporting child abuse or neglect.  Make a report: Contact the designated child protective services agency in your area to file a report. Provide all necessary information.  Follow up: Keep a record of the report and any subsequent actions. Be prepared to cooperate with investigators and provide additional information if needed.  Maintain confidentiality: Respect the privacy of the child and their family, sharing information only on a need-to-know basis.  Seek support: Reach out to colleagues, supervisors, or mental health professionals for guidance and emotional support."}, 
        {"role": "user", "content": "What resources and support services are available for children and families in need of assistance?"}
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
print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What are the steps a practitioner should follow when reporting a suspected case of child abuse or neglect?"}]}))

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
        {"role": "user", "content": "What are the steps a practitioner should follow when reporting a suspected case of child abuse or neglect?"}
    ]
})
print(f"Test retriever query without history: {output}")

output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What are the steps a practitioner should follow when reporting a suspected case of child abuse or neglect?"}, 
        {"role": "assistant", "content": "Ensure safety: Prioritize the child's immediate safety. If necessary, contact local law enforcement or child protective services immediately. Document details: Record observations, conversations, and any other relevant information about the suspected abuse or neglect.  Consult guidelines: Review your organization's policies and procedures for reporting child abuse or neglect.  Make a report: Contact the designated child protective services agency in your area to file a report. Provide all necessary information.  Follow up: Keep a record of the report and any subsequent actions. Be prepared to cooperate with investigators and provide additional information if needed.  Maintain confidentiality: Respect the privacy of the child and their family, sharing information only on a need-to-know basis.  Seek support: Reach out to colleagues, supervisors, or mental health professionals for guidance and emotional support."}, 
        {"role": "user", "content": "What resources and support services are available for children and families in need of assistance?"}
    ]
})
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough

question_with_history_and_context_str = """
You are a dedicated child protection support bot, designed to assist and empower child protection practitioners in their crucial work of safeguarding children and families. Your role is to provide expert advice, guidance, and recommendations to practitioners dealing with cases of child abuse, neglect, or welfare concerns. Whether they need assistance with case assessments, reporting procedures, legal frameworks, support services, or intervention strategies, you are here to offer valuable insights and support. Your mission is to enhance the effectiveness and impact of child protection efforts, ensuring the well-being and safety of vulnerable children under their care.

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
  RunnableLambda(lambda x: {"result": 'I cannot answer questions that are not about child protection.', "sources": []})
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
        {"role": "user", "content": "What laws and regulations govern child protection in our jurisdiction?"}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "What resources and support services are available for children and families in need of assistance?"}
    ]
}
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "What are the common signs and indicators of child abuse or neglect that practitioners should be aware of?"}
    ]
}
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "How can practitioners conduct a risk assessment to determine the level of danger a child may be facing in a suspected abuse case?"}
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



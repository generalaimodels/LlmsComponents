import sys
from pathlib import Path 
import logging 
from g4f.client import Client
sys.path.append(str(Path(__file__).resolve().parents[0]))

from agentdatacollection import (
    ConvertFolder_to_TxtFolder,
    AgentDataset,
    AgentDatasetLoader,
)
from agentdataretrieval import AgentContentRetrieval
from agenthistory import AgentHistoryManagerContentRetrieveUpdate
from langchain_huggingface import HuggingFaceEmbeddings
from agentcustommodel import (
    AgentModel,
    AgentPipeline,
    AgentPreProcessorPipeline,
    BitsAndBytesConfig,
    set_seed,
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/testing_agent.log',
    filemode='w'
)
logger = logging.getLogger(__name__)
Model_Name=""
cache_dir=""
model=AgentModel.load_model(
    model_type="causal_lm",
    model_name_or_path=Model_Name,
    cache_dir=cache_dir,
)
tokeiner=AgentPreProcessorPipeline(model_type="text",
                                   pretrained_model_name_or_path=Model_Name,
                                   cache_dir=cache_dir
                                   ).process_data()
model_name = "sentence-transformers/all-mpnet-base-v2" #@param {type:"string"} 
model_kwargs = {'device': 'cpu'}  #@param {type:"string"}
encode_kwargs = {'normalize_embeddings': False} #@param {type:"string"}
embedding_model= HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
logger.info(f"Agent is Started ...")
input_folder :str =[r"E:\LLMS\Fine-tuning\LlmsComponents\LLM_Components\AgentPipeline"]
output_folder :str =r"E:\LLMS\Fine-tuning\output"
ConvertFolder_to_TxtFolder(
    input_folders=input_folder,
    output_folder=output_folder
)
logger.info(f"Agent input folder {input_folder} outputdir {output_folder}")
agentdataset=AgentDataset(r"E:\LLMS\Fine-tuning\output").load_data()
files_names=list(agentdataset.keys())
logger.info(f"Only this files are selected for Agent to understanding content {files_names}")
agentdata=AgentDatasetLoader(dataset=agentdataset)
page_content_multi, metadata_multi=agentdata.load_multiple_datasets(dataset_names=files_names)
Agent_retriever=AgentContentRetrieval(embedding_model=embedding_model)
Agent_history=AgentHistoryManagerContentRetrieveUpdate(Agent_retriever.embeddings)
Agent_retriever.add_documents(page_content_multi, metadata_multi)
# # # Test single retrieval
query = "Exaplin great detail about AgentDatasetloader"
results = Agent_retriever.retrieve(query, top_k=4)
print(f"\nQuery: {query}")
for doc, score in results:
    print(f"Document: {doc.page_content}")
    print(f"Score: {score}")
    print(f"Metadata: {doc.metadata}")
    print()


query = "Exaplin great detail about AgentDatasetloader"
results = Agent_history.query(query, page_content_multi, metadata_multi )

for doc, score in results:
    print(f"Document: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Similarity Score: {score}")
    print("---")

client = Client()
prompt=f" write code in python for  Query: {query} content:{results[0][0].page_content} metadata {results[0][0].metadata}"
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
   
)
print(response.choices[0].message.content)
meta_data_llms={"rosponser":response.choices[0].message.content}
Agent_history.update_documents(new_documents=[response.choices[0].message.content],new_metadata=meta_data_llms)
query = "write python code for  AgentDatasetloader"
results = Agent_history.query(query, page_content_multi, metadata_multi )

for doc, score in results:
    print(f"Document: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Similarity Score: {score}")
    print("---")



prompt=f" write code in python for  Query: {query} content:{results[0][0].page_content} metadata {results[0][0].metadata}"
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
   
)
meta_data_llms={"rosponser":response.choices[0].message.content}
Agent_history.update_documents(new_documents=[response.choices[0].message.content],new_metadata=meta_data_llms)
results = Agent_history.query(query, page_content_multi, metadata_multi )


results = Agent_history.query(query, page_content_multi, metadata_multi )
prompt=f" write code in python for  Query: {query} content:{results[0][0].page_content} metadata {results[0][0].metadata}"
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
   
)
print(response.choices[0].message.content)
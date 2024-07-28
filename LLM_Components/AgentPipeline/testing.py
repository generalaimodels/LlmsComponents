import json
from agentdataset import AgentDataset,AgentDatasetLoader
from agentcontentretrieve import AgentContentRetrieval
from agenthistorysession import AgentHistorySession
from historyagentcontentretrieve import AgentHistoryManagerContentRetrieve
from agentprompting import AgentPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from agentchattemplete import AgentChatTokenizer
from agentgeneralisedmodel import AgentAIModel,AgentPreProcessorPipeline
model_name = "sentence-transformers/all-mpnet-base-v2" #@param {type:"string"} 
model_kwargs = {'device': 'cpu'}  #@param {type:"string"}
encode_kwargs = {'normalize_embeddings': False} #@param {type:"string"}
embedding_model= HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
Agent_dataset=AgentDataset(source=r"C:\Users\heman\Desktop\components\output").load_data()
print("Dataset::",Agent_dataset)
columns_name=list(Agent_dataset.keys())
print(list(Agent_dataset.keys()))  # List(str)  all files in the folder 
Agent_dataset_loader=AgentDatasetLoader(Agent_dataset) 
page_content, metadata = Agent_dataset_loader.load_dataset(columns_name[0]) # first/ one file is loading
page_content_multi, metadata_multi = Agent_dataset_loader.load_multiple_datasets(columns_name) # loading all files 
Agent_retriever=AgentContentRetrieval(embedding_model=embedding_model)
Agent_retriever.add_documents(page_content_multi, metadata_multi)
Agent_history=AgentHistoryManagerContentRetrieve(embedding_model)
# # Test single retrieval
query = "Linux terminal"
results = Agent_retriever.retrieve(query, top_k=1)
print(f"\nQuery: {query}")
for doc, score in results:
    print(f"Document: {doc.page_content}")
    print(f"Score: {score}")
    print(f"Metadata: {doc.metadata}")
    print()
    
# Test batch retrieval
queries = [
    "Career Counselor",
    "Synonym finder",
    "Commit Message Generator"
]
batch_results = Agent_retriever.batch_retrieve(queries, top_k=1)
for i, query_results in enumerate(batch_results):
    print(f"\nQuery: {queries[i]}")
    for doc, score in query_results:
        print(f"Document: {doc.page_content}")
        print(f"Score: {score}")
        print(f"Metadata: {doc.metadata}")
        print()

query = "Product Manager"
results = Agent_history.query(query, page_content_multi, metadata_multi )
for doc, score in results:
    print(f"Document: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Similarity Score: {score}")
    print("---")
    
template = AgentPromptTemplate(
    role="Advanced AI Agents will use that information, content, metadata",
    content="Your aim is to explain details about Query:\n '{query}' by looking at details in Content:\n '{content}' and Metadata:\n '{metadata}'",
    parameters={
        "query": query,
        "content": results[0][0].page_content,
        "metadata": results[0][0].metadata
    },
    example="Responding to your query by looking into content.",
    constraints=[
        "It should look like generated content.",
        "Don't provide any harmful content.",
        "Response should be crisp."
    ],
    output_format="Here is the detailed response to the query..."
)
formatted_prompt = template.format()
agent_input = json.dumps(formatted_prompt, indent=2)
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Replace with your preferred model
chat_tokenizer = AgentChatTokenizer(model_name, quantization=None,cache_dir=r"C:\Users\heman\Desktop\components\model")
Agent_response = chat_tokenizer.generate_response(agent_input, max_length=512,)
print(Agent_response)

Model_Name_or_Path="meta-llama/Meta-Llama-3.1-8B-Instruct"
CAUSAL_LM="causal_lm"
CACHE_DIR=""
model=AgentAIModel.load_model(              # Model is loading
    model_type=CAUSAL_LM,
    model_name_or_path=Model_Name_or_Path,
    cache_dir=CACHE_DIR,
)

tokenizer=AgentPreProcessorPipeline(           # Tokenizer is loading
    model_type="text",
    pretrained_model_name_or_path=Model_Name_or_Path,
    cache_dir=CACHE_DIR,
).process_data()

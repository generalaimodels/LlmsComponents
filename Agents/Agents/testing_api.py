from Agent_dataset import SampleDataset123, AdvancedDatasetLoader
from content_retrieve import AdvancedVectorRetrieval
from History_manger import AdvancedHistoryManager_upate


if __name__ == "__main__":
    dataset=SampleDataset123(source=r"C:\Users\heman\Desktop\components\output").load_data()
    
    loader = AdvancedDatasetLoader(dataset)
    page_content, metadata = loader.load_dataset('prompts')
    print("Single dataset load:")
    print(f"Page content: {page_content[0]}")
    print(f"Metadata: {metadata[0]}")
    retriever = AdvancedVectorRetrieval()
    retriever.add_documents(page_content, metadata)
    query = "linux terminal "
    results = retriever.retrieve(query, top_k=1)
    print(f"\nQuery: {query}")
    print(f"Document: {results[0][0].page_content}")
    history_manager=AdvancedHistoryManager_upate(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    query = "linux terminal"
    
    results = history_manager.query(query, page_content, metadata)
    
    for doc, score in results:
        print(f"Document: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print(f"Similarity Score: {score}")
        print("---")
    # # Load multiple datasets
    # page_content, metadata = loader.load_multiple_datasets(['prompts', 'read', 'testing'])
    # print(f"Page content: {page_content[10 :11]}")
    # print(f"Metadata: {metadata[10:11]}")
    

    
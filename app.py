from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

# Step 1: Load PDF
loader = PyPDFLoader("data/solar_system.pdf")
documents = loader.load()
print("PDF Loaded")

# Step 2: Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)
print("Text Split into chunks")

# Step 3: Fake Embeddings (No torch needed)
embeddings = FakeEmbeddings(size=384)

# Step 4: Create vector store
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

print("Vector Database Created")

# Step 5: Simple chatbot (no API needed)
print("\nChatbot Ready! Type 'exit' to stop.\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    results = retriever.get_relevant_documents(query)

    print("\nBot:")
    for doc in results[:2]:
        print(doc.page_content[:300])
    print("\n" + "-"*50)
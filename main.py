from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA


def main():
    print("Iniciando o processo de RAG...")
    
    load_dotenv()
    pdf_path = "mitologia-grega.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    persist_dir = "vectorstore/manual_chroma"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    vectorstore.persist()
    print(f"Persistido em: {persist_dir}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    perguntas = [
        "Qual a história de Zeus?",
        "Qual a importância do olimpo na mitologia grega?",
        "Quais são os principais deuses na mitologia grega?",
    ]

    for q in perguntas:
        result = qa_chain({"query": q})
        print("\nPergunta:", q)
        print("Resposta:", result["result"])
        print("Fontes:")
        for i, d in enumerate(result["source_documents"], 1):
            meta = d.metadata
            preview = d.page_content[:180].replace("\n", " ")
            print(f"  {i}. page={meta.get('page')}, source={meta.get('source')}, trecho='{preview}...'")


if __name__ == "__main__":
    main()

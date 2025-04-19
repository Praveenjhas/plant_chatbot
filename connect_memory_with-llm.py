import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# ✅ Load environment variables
load_dotenv()

# ✅ Paths
DB_FAISS_PATH = "vectorstore/db_faiss"

# ✅ Memory (last exchange only)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=1,
    return_messages=True,
    output_key="answer"
)

# ✅ LLM setup (OpenRouter API with Mistral-7B)
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    temperature=0.5,
    max_tokens=512,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ✅ Prompt template for HyDE
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="Please write a detailed answer to the following question:\n\n{question}"
)

# ✅ LLMChain for HyDE
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# ✅ Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ HyDE wrapper around base embeddings
hyde_embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=embedding_model
)

# ✅ Load FAISS vector store
vectorstore = FAISS.load_local(
    DB_FAISS_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# ✅ Retriever with HyDE
retriever = vectorstore.as_retriever(embedding=hyde_embeddings)

# ✅ QA Chain with memory and source documents
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# ✅ Start chat loop
print("\n🤖 Chatbot is ready. Ask your questions!\n(Type 'exit' to stop)\n")

try:
    while True:
        user_query = input("🧑 You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting. See you next time!")
            break

        response = qa_chain.invoke({"question": user_query})

        print("\n🤖 Bot:")
        print(response["answer"])

        print("\n📚 Sources:")
        for doc in response["source_documents"]:
            print("-", doc.metadata.get("source", "Unknown"))
        print("\n" + "-"*60 + "\n")

except KeyboardInterrupt:
    print("\n👋 Exiting due to keyboard interrupt.")

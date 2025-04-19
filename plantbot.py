import os
import io
import re
import json
import time
import uuid
import streamlit as st
from gtts import gTTS
from datetime import datetime
from langdetect import detect
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

DB_FAISS_PATH = "vectorstore/db_faiss"

# ----------------- UTILITY -----------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def load_llm_openrouter():
    try:
        # Verify the key exists first
        if "openrouter" not in st.secrets or not st.secrets["openrouter"].get("api_key"):
            st.error("OpenRouter API key not configured. Please check your secrets.")
            st.stop()

        return ChatOpenAI(
            model="mistralai/mistral-7b-instruct",
            temperature=0.5,
            max_tokens=512,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=st.secrets["openrouter"]["api_key"],
            headers={
                "HTTP-Referer": "https://mysteramlitchatbot.streamlit.app/",  # Replace with your actual URL
                "X-Title": "PlantBot Chatbot"
            }
        )
    except Exception as e:
        st.error(f"LLM initialization failed: {str(e)}")
        st.stop()

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def text_to_speech(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

def extract_sources(source_documents):
    sources = set()
    for doc in source_documents:
        if 'source' in doc.metadata:
            sources.add(doc.metadata['source'])
    return list(sources)

def format_sources(sources):
    if not sources:
        return ""
    lines = ["\n**Sources:**"]
    for idx, src in enumerate(sources, 1):
        lines.append(f"{idx}. [{src}]({src})")
    return "\n".join(lines)

def highlight_keywords(text):
    keywords = ["strawberry", "disease", "soil", "climate", "water", "fertilizer"]
    for kw in keywords:
        text = re.sub(fr"\b({kw})\b", r"**\1**", text, flags=re.IGNORECASE)
    return text

# ----------------- MAIN APP -----------------
def main():
    st.set_page_config(page_title="Strawberry Chatbot 🍓", layout="centered")
    st.markdown(custom_css, unsafe_allow_html=True)
    st.title("🍓 Strawberry Chatbot – Ask Your Questions!")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = {}

    # Display chat messages from history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "🧑"):
            st.markdown(message["content"])
            
            if message["role"] == "assistant":
                audio_key = f"audio_{idx}"
                btn_key = f"btn_{idx}"
                
                # Always show the Generate Audio button
                if st.button("🔊 Generate Audio", key=btn_key):
                    with st.spinner("Generating audio..."):
                        audio_data = text_to_speech(message["content"], message.get("lang", "en"))
                        st.session_state.audio_cache[audio_key] = audio_data
                        st.rerun()
                
                # Show audio player if audio exists in cache
                if audio_key in st.session_state.audio_cache:
                    st.audio(st.session_state.audio_cache[audio_key], format="audio/mp3")

    prompt = st.chat_input("Ask something about strawberries...")

    if prompt:
        # Save user's message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown("_✍️ Thinking..._")

            # Generate answer
            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm_openrouter(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PromptTemplate(
                    template=""" 
Answer the user's question using the context provided below and the knowledge that you have for that plant use that also.
If the context contains relevant information, use it directly. If the context is insufficient, include your own knowledge but make sure it's relevant. Cite the sources you have taken to answer.

Context:
{context}

Question:
{question}

Answer:
""",
                    input_variables=["context", "question"]
                )} 
            )

            response = qa_chain.invoke({"query": prompt})
            answer = highlight_keywords(response["result"])
            sources = format_sources(extract_sources(response["source_documents"]))
            full_answer = answer + "\n\n" + sources

            # Typewriter display
            displayed_text = ""
            for char in full_answer:
                displayed_text += char
                placeholder.markdown(displayed_text + "▌")
                time.sleep(0.008)
            placeholder.markdown(displayed_text)

            # Save assistant's message
            lang_code = detect_language(full_answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "lang": lang_code
            })
            
            # Immediately show Generate Audio button for this new message
            new_audio_key = f"audio_{len(st.session_state.messages)-1}"
            new_btn_key = f"btn_{len(st.session_state.messages)-1}"
            
            if st.button("🔊 Generate Audio", key=new_btn_key):
                with st.spinner("Generating audio..."):
                    audio_data = text_to_speech(full_answer, lang_code)
                    st.session_state.audio_cache[new_audio_key] = audio_data
                    st.rerun()
            
            if new_audio_key in st.session_state.audio_cache:
                st.audio(st.session_state.audio_cache[new_audio_key], format="audio/mp3")

    # Sidebar
    with st.sidebar:
        st.header("🛠️ Tools")
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.session_state.audio_cache = {}
            st.rerun()

        st.download_button(
            label="📄 Download .txt",
            data="\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

        st.download_button(
            label="🧾 Download .json",
            data=json.dumps(st.session_state.messages, indent=2),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# ----------------- CSS -----------------
custom_css = """
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }

    .stMarkdown {
        font-size: 16px;
        line-height: 1.7;
    }

    .stChatInput input {
        padding: 12px;
        font-size: 16px;
        border-radius: 12px !important;
        border: 1px solid #ccc;
    }

    .stButton > button {
        border-radius: 10px;
        padding: 6px 12px;
        font-weight: bold;
    }

    audio {
        width: 100%;
        margin-top: 10px;
    }
</style>
"""

if __name__ == "__main__":
    main()

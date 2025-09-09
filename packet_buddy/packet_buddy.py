import os
import json
import requests
import subprocess
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import TextLoader

# Message classes
class Message:
    def __init__(self, content):
        self.content = content

class HumanMessage(Message):
    """Represents a message from the user."""
    pass

class AIMessage(Message):
    """Represents a message from the AI."""
    pass

@st.cache_resource
def load_model():
    with st.spinner("Downloading Instructor XL Embeddings Model locally....please be patient"):
        embedding_model=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"})
    return embedding_model

# Function to generate priming text based on pcap data

def returnSystemText(pcap_data: str) -> str:
    PACKET_WHISPERER = f"""
    You are a helper assistant specialized in analysing packet captures.

    Use the packet_capture_info provided below plus any retrieved context
    to answer all questions truthfully. Always reference frame numbers
    or timestamps if possible.

    ### Protocol hints
    HCI packets always start with 0x5A 0x0F, contain the protocol tag 0xAB 0xBA 0xCE 0xDE, and end with 0x2E 0x8D

    ### packet_capture_info
    {pcap_data}
    """
    return PACKET_WHISPERER


# Define a class for chatting with pcap data
class ChatWithPCAP:
    def __init__(self, json_path, extra_text_path=None):
        self.embedding_model = load_model()
        self.json_path = json_path
        self.extra_text_path = extra_text_path
        self.conversation_history = []
        self.load_json()
        if self.extra_text_path:
            self.load_text_doc()
        self.split_into_chunks()
        self.store_in_chroma()
        self.setup_conversation_memory()
        self.setup_conversation_retrieval_chain()
        self.priming_text = self.generate_priming_text()

    def load_json(self):
        self.loader = JSONLoader(
            file_path=self.json_path,
            jq_schema=".[] | ._source.layers",
            text_content=False
        )
        self.pages = self.loader.load_and_split()

    def load_text_doc(self):
        """Load an additional explanatory text document."""
        loader = TextLoader(self.extra_text_path)
        extra_docs = loader.load()
        self.pages.extend(extra_docs)

    def split_into_chunks(self):
        with st.spinner("Splitting into chunks..."):
            self.text_splitter = SemanticChunker(self.embedding_model)
            self.docs = self.text_splitter.split_documents(self.pages)

    def store_in_chroma(self):
        with st.spinner("Storing in Chroma..."):
            # Now, pass this wrapper to Chroma.from_documents
            self.vectordb = Chroma.from_documents(self.docs, self.embedding_model)
            self.vectordb.persist()

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",   # <- tell memory what the input field is
            output_key="answer",    # <- tell memory what to save from outputs
            return_messages=True
        )

    def setup_conversation_retrieval_chain(self):
        self.llm = Ollama(
            model=st.session_state['selected_model'],
            base_url="http://ollama:11434",
            num_ctx=32768   # 👈 pass directly here
        )
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectordb.as_retriever(search_kwargs={"k": 10}),
            memory=self.memory,
            chain_type="map_reduce", # or stuff or map_rerank
            return_source_documents=True
        )

    def generate_priming_text(self):
        # Take more than 5 pages if available, but keep it compact
        raw_preview = " ".join([d.page_content[:300] for d in self.pages[:20]])

        # Build a summary-like snippet
        pcap_summary = f"Here are some excerpts from the packet capture:\n{raw_preview}\n\n"

        # Combine with PACKET_WHISPERER instructions
        return returnSystemText(pcap_summary)


    def chat(self, question):
        primed_question = self.priming_text + "\n\n" + question  # (see Option B note below)
        response = self.qa.invoke({"question": primed_question})
        if response:
            # Synthesized (reduce) answer:
            synthesized = response.get("answer", "")

            # “Raw” material: show retrieved docs that fed the map step.
            # (LangChain doesn’t expose the per-doc map outputs directly,
            # but you can at least show the source chunks and/or re-run map prompts.)
            sources = response.get("source_documents", [])

            st.write("Query:", primed_question)
            st.markdown("**Synthesized Answer (Map–Reduce):**")
            st.markdown(synthesized)

            if sources:
                st.markdown("**Raw Inputs (Mapped Sources):**")
                for i, d in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:**")
                    st.code(d.page_content[:1200] + ("..." if len(d.page_content) > 1200 else ""))

            return {"answer": synthesized}
   
# Function to convert pcap to JSON
def pcap_to_json(pcap_path, json_path):
    command = f'tshark -nlr {pcap_path} -T json > {json_path}'
    subprocess.run(command, shell=True)

def get_ollama_models(base_url):
    try:       
        response = requests.get(f"{base_url}api/tags")  # Corrected endpoint
        response.raise_for_status()
        models_data = response.json()
        
        # Extract just the model names for the dropdown
        models = [model['name'] for model in models_data.get('models', [])]
        return models
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get models from Ollama: {e}")
        return []

# Streamlit UI for uploading and converting pcap file
def upload_and_convert_pcap():
    st.title('Packet Buddy - Chat with Packet Captures')
    uploaded_file = st.file_uploader("Choose a PCAP file", type="pcap")
    extra_doc = st.file_uploader("Optionally, upload an explanatory text file", type=["txt", "md"])

    if uploaded_file:
        if not os.path.exists('temp'):
            os.makedirs('temp')
        pcap_path = os.path.join("temp", uploaded_file.name)
        json_path = pcap_path + ".json"
        with open(pcap_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        pcap_to_json(pcap_path, json_path)

        # Save extra text file if provided
        extra_text_path = None
        if extra_doc:
            extra_text_path = os.path.join("temp", extra_doc.name)
            with open(extra_text_path, "wb") as f:
                f.write(extra_doc.getvalue())

        # Store in session state
        st.session_state['json_path'] = json_path
        st.session_state['extra_text_path'] = extra_text_path
        st.success("Files uploaded and converted.")

        # Fetch Ollama models
        models = get_ollama_models("http://ollama:11434/")
        if models:
            selected_model = st.selectbox("Select Model", models)
            st.session_state['selected_model'] = selected_model
            
            if st.button("Proceed to Chat"):
                st.session_state['page'] = 2      

# Streamlit UI for chat interface
def chat_interface():
    st.title('Packet Buddy - Chat with Packet Captures')

    json_path = st.session_state.get('json_path')
    if not json_path or not os.path.exists(json_path):
        st.error("PCAP file missing or not converted. Please go back and upload a PCAP file.")
        return

    # Create / reuse the chat engine
    if 'chat_instance' not in st.session_state:
        st.session_state['chat_instance'] = ChatWithPCAP(
            json_path=json_path,
            extra_text_path=st.session_state.get('extra_text_path')
        )

    # Create message store
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Render past messages
    for m in st.session_state['messages']:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Persistent chat input (like ChatGPT)
    prompt = st.chat_input("Ask a question about the PCAP data…")
    if prompt:
        # Show the user bubble immediately
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer from your chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = st.session_state['chat_instance'].chat(prompt) or {}
                answer = result.get("answer", "I couldn't find a specific answer.")
                st.markdown(answer)

        # Store assistant reply for re-render
        st.session_state['messages'].append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state['page'] = 1

    if st.session_state['page'] == 1:
        upload_and_convert_pcap()

    elif st.session_state['page'] == 2:
        chat_interface()

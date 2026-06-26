import streamlit as st
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(
    page_title="CampusAssistAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------- CSS ----------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(-45deg,#020024,#090979,#4B0082,#00d4ff);
    background-size:400% 400%;
    animation:bg 15s ease infinite;
}

@keyframes bg{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

.title{
    font-size:65px;
    font-weight:bold;
    text-align:center;
    background:linear-gradient(90deg,#00F5FF,#FF00FF,#FFD700);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:glow 2s infinite alternate;
}

@keyframes glow{
from{
text-shadow:0 0 10px cyan;
}
to{
text-shadow:0 0 35px magenta;
}
}

.subtitle{
    text-align:center;
    font-size:22px;
    color:white;
    margin-bottom:20px;
}

.response{
    background:rgba(255,255,255,0.15);
    padding:20px;
    border-radius:20px;
    color:white;
    margin-top:20px;
}

.stTextInput>div>div>input{
    border-radius:15px;
    border:2px solid cyan;
}

</style>
""", unsafe_allow_html=True)

# ---------- Heading ----------


st.markdown(
'<div class="title">🎓 CampusAssistAI ✨</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">💜 Your Smart AI Campus Assistant 💜</div>',
unsafe_allow_html=True
)

st.image("hero.png", use_container_width=True)

# ---------- Load Vector DB ----------

embeddings = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
"vectorstore",
embeddings,
allow_dangerous_deserialization=True
)

# ---------- Gemini ----------

llm = ChatGoogleGenerativeAI(
model="gemini-2.5-flash",
google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ---------- Chat ----------

question = st.text_input(
"💬 Ask your campus question"
)

if question:

    docs = db.similarity_search(question,k=3)

    context="\n\n".join([doc.page_content for doc in docs])

    prompt=f"""
You are CampusAssistAI.

Context:
{context}

Question:
{question}

Answer in a friendly way.
"""

    with st.spinner("🤖 Thinking..."):

        response=llm.invoke(prompt)

    st.markdown(
    f"""
<div class="response">

### 🤖 CampusAssistAI

{response.content}

</div>
""",
unsafe_allow_html=True
)
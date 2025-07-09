import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import os
import tempfile
import hashlib
import warnings
from PIL import Image
from datetime import datetime
import uuid
import csv
import io
import time

# Policy module imports
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema.document import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
#from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import io
import json

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.schema import HumanMessage, SystemMessage

# Audit module imports
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from chatbot_utils import (
    get_metadata_from_mysql,
    create_vector_db_from_metadata,
    retrieve_top_tables,
    create_llm_table_retriever,
    question_reframer,
    generate_sql_query_for_retrieved_tables,
    execute_sql_query,
    analyze_sql_query,
    finetune_conv_answer,
    debug_query,
)

warnings.filterwarnings("ignore")

OPENAI_KEY       = st.secrets["openai"]["api_key"]
DB_USER          = st.secrets["mysql"]["user"]
DB_PASSWORD      = st.secrets["mysql"]["password"]
DB_HOST          = st.secrets["mysql"]["host"]
DB_PORT          = st.secrets["mysql"]["port"]
DB_NAME          = st.secrets["mysql"]["database"]
NVIDIA_API_KEY   = st.secrets["nvidia"]["api_key"]



# -- Configurations --
logo = Image.open(r"Assets/aurex_logo.png")
descriptions_file = r"Assets/all_table_metadata_v4.txt"
examples_file = r"Assets/Example question datasets.xlsx"

db_config = {
    "user": DB_USER,
    "password": DB_PASSWORD ,
    "host": DB_HOST,
    "port": DB_PORT,
    "database": DB_NAME
}


scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
# Convert st.secrets to a JSON-style dict
creds_dict = dict(st.secrets["gsheets"])
# Convert to actual JSON string and parse it
creds_json = json.loads(json.dumps(creds_dict))
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json,scope)
client = gspread.authorize(creds)
sheet = client.open("Streamlit_Chatbot_Logs").sheet1  

headers = ["session_id","question_id","timestamp","question","sql_query",
"conversational_answer","rating", "comments"]


st.set_page_config(initial_sidebar_state='expanded')
st.image(logo, width=150)
st.title("Welcome to Aurex AI Chatbot")
policy_flag = st.toggle("DocAI")

# 2. Sidebar expander for intermediate steps
with st.sidebar:
    st.markdown("### ⚙️ Intermediate Steps")
    steps_expander = st.expander("Show steps", expanded=False)
    step_titles = ["Reframed Question with memory",
        "Top 10 Tables",
        "Top 3 Tables via LLM",
        "Reframed Question",
        "Generated SQL",
        "Debugged SQL",
        "Query Result Sample",
        "Initial Conversational Draft"
    ]
    placeholders = {title: steps_expander.container() for title in step_titles}


class PrintRetrievalHandler(BaseCallbackHandler):
        def __init__(self, container):
            self.status = container.status("**Context Retrieval**")

        def on_retriever_start(self, serialized: dict, query: str, **kwargs):
            self.status.write(f"**Question:** {query}")
            self.status.update(label=f"**Context Retrieval:** {query}")

        def on_retriever_end(self, documents, **kwargs):
            for idx, doc in enumerate(documents):
                source = os.path.basename(doc.metadata["source"])
                self.status.write(f"**Document {idx} from {source}**")
                self.status.markdown(doc.page_content)
            self.status.update(state="complete")

def serialize_chat_history():
    """
    Convert StreamlitChatMessageHistory into a plain-text string.
    Each line is "User: …" or "Assistant: …" in chronological order.
    """
    messages = st.session_state.risk_chat_history.messages  # list of ChatMessage
    lines = []
    for m in messages:
        role = "User" if m.type == "human" else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


# Chart file hash (not used directly here)
def checkfilechange(file_path):
    with open(file_path, "rb") as f:
        newhash = hashlib.md5(f.read()).hexdigest()
    return newhash


# CSV logger
def log_csv(entry):
    log_file = "chat_log.csv"
    write_header = not os.path.exists(log_file)
    with open(log_file, "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=entry.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(entry)


def log_to_google_sheets(entry):
    """
    Appends a dictionary entry as a new row in the Google Sheet.
    """
    # Map the entry to the headers
    row = [
        entry.get("session_id", ""),
        entry.get("question_id", ""),
        entry.get("timestamp", ""),
        entry.get("question", ""),
        entry.get("sql_query", ""),
        entry.get("conversational_answer", ""),
        entry.get("rating", ""),
        entry.get("comments", "")
    ]
    
    # Append the row to the Google Sheet
    sheet.append_row(row, value_input_option="USER_ENTERED")


# Core processing, without UI
def process_risk_query(llm, user_question):
    # Check if 'conn' and 'vector_store' are already in session state
    if 'conn' not in st.session_state or 'vector_store' not in st.session_state:
        with st.spinner("🔍 Connecting to the Risk management database..."):
            # Establish the database connection and create the vector store
            conn, metadata = get_metadata_from_mysql(db_config, descriptions_file=descriptions_file)
        with st.spinner("🔍 Connecting to the vector database..."):
            vector_store = create_vector_db_from_metadata(metadata)
            # Store them in session state
            st.session_state.conn = conn
            st.session_state.metadata = metadata
            st.session_state.vector_store = vector_store
    else:
        # Retrieve from session state
        conn = st.session_state.conn
        metadata = st.session_state.metadata
        vector_store = st.session_state.vector_store
        
    if conn is None or not metadata:
            return "Sorry, I was not able to connect to Database", None, ""
    with st.spinner("📊 Retrieving the metadata for most relevant tables..."):
        docs = retrieve_top_tables(vector_store, user_question, k=10)
        top_names = [d.metadata["table_name"] for d in docs]
        placeholders["Top 10 Tables"].markdown("## Top 10 Tables after stage 1 retrieval")
        placeholders["Top 10 Tables"].write(", ".join(top_names))
        example_df = pd.read_excel(examples_file)
        top3 = create_llm_table_retriever(llm, user_question, top_names, example_df)
        placeholders["Top 3 Tables via LLM"].markdown("## Top 3 Tables after stage 2 retrieval")
        placeholders["Top 3 Tables via LLM"].write(top3)
        filtered = [d for d in docs if d.metadata["table_name"] in top3]

    with st.spinner("📝 Reframing question based on metadata..."):
        reframed = question_reframer(filtered, user_question, llm)
        placeholders["Reframed Question"].markdown("## Question Rephrasing Process")
        placeholders["Reframed Question"].write(reframed)

    with st.spinner("🛠️ Generating SQL query..."):
        sql = generate_sql_query_for_retrieved_tables(filtered, reframed, example_df, llm)
        placeholders["Generated SQL"].markdown("## SQL Query Generation Process")
        placeholders["Generated SQL"].code(sql)
        
    with st.spinner("🚀 Executing SQL query..."):
        result, error = execute_sql_query(conn, sql)
        if result is None or result.empty:
            with st.spinner("🧪 Debugging SQL query..."):
                sql = debug_query(filtered, user_question, sql, llm, error)
                result, error = execute_sql_query(conn, sql)
                placeholders["Debugged SQL"].markdown("## SQL Query Debugging Process")
                placeholders["Debugged SQL"].code(sql)
            if result is None or result.empty:
                return "Sorry, I couldn't answer your question.", None, sql
        placeholders["Query Result Sample"].markdown("## Tabular Result of SQL Query")        
        #placeholders["Query Result Sample"].table(result)
        try:
            placeholders["Query Result Sample"].dataframe(result, width=600, height=300)
        except ValueError as e:
            # detect and drop duplicate columns
            if "Duplicate column names found" in str(e):
                result = result.loc[:, ~result.columns.duplicated()]
                placeholders["Query Result Sample"].dataframe(result, width=600, height=300)
            else:
                return "Sorry, I couldn't answer your question.", None, sql
       

    with st.spinner("📈 Analyzing SQL query results..."):
        conv = analyze_sql_query(user_question, result.to_dict(orient='records'), llm)
        placeholders["Initial Conversational Draft"].markdown("## Initial Answer before finetuning process")
        placeholders["Initial Conversational Draft"].write(conv)

    with st.spinner("💬 Finetuning conversational answer..."):
        conv = finetune_conv_answer(user_question, conv, llm)

    return conv, result, sql



def is_followup_question(llm, memory, current_question):
    """
    Use an LLM to determine if the current question is a follow-up to the last Q&A in memory.

    Returns:
        bool: True if it's a follow-up, False otherwise.
    """
    chat_history = "\n".join([f"user: {entry['content']}" for entry in memory])
    # Prepare prompt template
    followup_prompt = PromptTemplate(input_variables=["chat_history", "question"],
        template = """You are a follow‑up detection assistant. Your job is to decide whether the user’s latest question is a direct continuation of the prior dialogue.

        Chat History:
        {chat_history}
        
        New user question:
        {question}
        
        Instructions:
        1. If the new question explicitly builds on or refers back to a specific element in the history (e.g. uses pronouns like “those,” “that region,” mentions the same unresolved entity, or asks for “further breakdown” of a previously requested metric), answer “Yes.”
        2. If it is a standalone request—even if it’s topically similar (same domain or metrics) but does not depend on a prior answer—answer “No.”
        3. Check for clear signals of dependency:
               - Referential pronouns or phrases (“that,” “those,” “further,” “next,” “again”) pointing to a past result.
               - Questions asking “how many of the above,” “among those,” “build on the previous result,” etc.
        4. Do NOT treat mere topic overlap (e.g. “sales,” “risks,” “tickets”) as a follow‑up—there must be an explicit link back to a specific prior response.
        5. Do NOT hallucinate or infer any context beyond what’s literally in the history.
        6. Do NOT provide any additional text—only “Yes” or “No.”
        
        Examples:
        • History:  
          List the total sales by region for Q1.  
          Break down Q1 sales by product category.  
          Which category in the Northeast had the highest growth?
        
          Question: What were the top three best‑selling products in that region? → Yes
        
        • History:  
          Show the count of support tickets by priority level.  
          List all tickets escalated to Level 2 in May.  
          How many of the Level 2 tickets were resolved within SLA?
        
          Question: Generate a monthly trend chart for new tickets in June. → No

        • History:
          Retrieve monthly revenue for January.
          Show the same for February.
          Compare January and February revenue.
            
          Question: What was the percentage change between those two months? → Yes
            
        • History:
          List all active projects in Q2.
          Filter to those with budgets over $100K.
          Sort by expected ROI.
            
          Question: “List all active projects in Q3.” → No
            
        • History:
          What were our top five selling products last quarter?
          Drill down sales by product category.
          Highlight categories with under 10% growth.
            
          Question: Which specific product in the above under-performing categories needs restocking? → Yes
            
        • History:
          How many new user sign-ups did we get in May?
          Break that down by referral source.
          What was the conversion rate from email campaigns?
            
          Question: What is the user churn rate in August? → No
                    
          Respond now with “Yes” or “No” only:""")


    chain = LLMChain(llm=llm,prompt=followup_prompt, verbose=True )
    result = chain.run(question=current_question,chat_history=chat_history).strip().lower() 
       

    return result.startswith("y")


#This function is not used currently anywhere plaease ignore this function
#def rephrase_question_with_memory(llm, memory, current_question)
    #rephrase_prompt = PromptTemplate(input_variables=["chat_history", "question"],
        #template="""
        #Given the following conversation history and a follow-up question, rephrase the question to be a standalone query.
        
        #Chat History:
        #{chat_history}
        
        #Follow-up question:
        #{question}
        
        #Standalone question:""".strip() )

    #chain = LLMChain(llm=llm,prompt=rephrase_prompt,memory=memory, verbose=False )
    #standalone_qstn = chain.run(question=current_question).strip()                 

    #return standalone_qstn



# -- Policy Module --
if policy_flag:
    st.success("Connected to Policy Module")
    uploaded = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if not uploaded:
        st.info("Please upload PDF documents to continue.")
        st.stop()
        
    def configure_retriever(files):
        temp = tempfile.TemporaryDirectory()
        docs = []
        for f in files:
            path = os.path.join(temp.name, f.name)
            with open(path, "wb") as out:
                out.write(f.getvalue())
            docs.extend(PyPDFLoader(path).load())
        splits = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(docs)
        emb = OpenAIEmbeddings(model="text-embedding-3-large", api_key= OPENAI_KEY)
        db = DocArrayInMemorySearch.from_documents(splits, emb)
        return db.as_retriever(search_type="mmr", search_kwargs={"k":2, "fetch_k":4})
    
    with st.spinner("Loading and processing documents..."):
        retriever = configure_retriever(uploaded)
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)
        #llm_policy = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key= OPENAI_KEY , temperature=0, streaming=True)
        llm_policy= ollama_llm = ChatOllama(model="llama3",temperature=0,base_url="https://cf8d-34-60-249-53.ngrok-free.app")
        qa_chain = ConversationalRetrievalChain.from_llm(llm_policy, retriever=retriever, memory=memory, verbose=False)
    
    if len(msgs.messages)==0 or st.sidebar.button("Clear history"):
        msgs.clear(); msgs.add_ai_message("How can I help you?")
        
    for m in msgs.messages:
        st.chat_message("user" if m.type=="human" else "assistant").write(m.content)
        
    if prompt := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(prompt)
        with st.spinner("Generating policy response..."):   
            handler = BaseCallbackHandler()
            retrieval_handler = PrintRetrievalHandler(st.container())
            resp = qa_chain.run(prompt, callbacks=[handler, retrieval_handler])
        with st.chat_message("assistant"):
            st.write(resp)

# -- Risk/Audit Module --
else:
    st.success("Connected to Risk Management Module")

    # ──────────────────────────────────────────────────────────────
    # 1) Session‐scoped state: session_id, chat_history, and memory
    # ──────────────────────────────────────────────────────────────
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if 'risk_msgs' not in st.session_state:
        # risk_msgs is a list of dicts like {"role": "user"/"assistant", "content": "..."}
        st.session_state.risk_msgs = []

    if 'risk_mem' not in st.session_state:
        # risk_mem is a list of dicts like {"role": "user"/"assistant", "content": "..."}
        st.session_state.risk_mem = []

    # ──────────────────────────────────────────────────────────────
    # 2) Initialize LangChain LLM (you use ChatNVIDIA; here is ChatOpenAI)
    # ──────────────────────────────────────────────────────────────
    #llm_audit = ChatNVIDIA(model="meta/llama-3.3-70b-instruct",api_key= NVIDIA_API_KEY,temperature=0, num_ctx=50000)
    llm_audit = ChatNVIDIA(model="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",base_url="http://54.161.46.7/v1/",temperature=0,max_tokens=1024, top_p=0.1,seed=42)
    
    # ──────────────────────────────────────────────────────────────
    # 3) Set up LangGraph short‐term memory (thread‐scoped InMemoryStore)
    #    and wrap it into a minimal “memory agent” using create_react_agent.
    # ──────────────────────────────────────────────────────────────
    #
    # 3.1) Create an in‐memory store where LangGraph will persist state.
    checkpointer = InMemorySaver()
    #
    # 3.3) Build a “zero‐tool ReAct agent” whose only job is to rephrase or
    #      pass through the question.”
    memory_agent_prompt = """You are a memory-aware assistant specialized in short-term conversational context.

        Input: A chat history (containing user and assistant turns) and a new user message.
        
        Task:
        1. If the new user message is a follow-up that relies on prior context, rephrase it into a fully self-contained question by incorporating all necessary details from the chat history.
        2. If the message is standalone or the first in the conversation, return it unchanged.
        3. Please do not hallucinate. 
        4. Please be very specific while framing question and keep the question short and brief.
        5. Please decide accurately if current question is a followup or not before rephrasing and in case it is a standalone please avoid rephrasing.
        
        Output strictly the final rephrased or original question—no extra explanations, comments, or formatting. """
        

    memory_agent = create_react_agent(
        model=llm_audit,
        tools=[],  # no tools needed; agent only does “rewrite”
        prompt=memory_agent_prompt,
        checkpointer=checkpointer,  # this enables short-term memory
    )
    #
    # Note: By default, create_react_agent stores its entire “state” (the
    # list of messages + scratchpad) under the given `thread_id`.
    # :contentReference[oaicite:0]{index=0}

    # ──────────────────────────────────────────────────────────────
    # 4) Display any previous chat‐turns in Streamlit
    # ──────────────────────────────────────────────────────────────
    for msg in st.session_state.risk_msgs:
        st.chat_message(msg['role']).write(msg['content'])

    # ──────────────────────────────────────────────────────────────
    # 5) When the user types a new prompt:
    # ──────────────────────────────────────────────────────────────
    if prompt := st.chat_input(placeholder="Ask a question about the Risk Management module"):
        start_time = time.time()
   
        # 5.1) Show the user message in the UI
        st.chat_message("user").write(prompt)
        followup_flag = is_followup_question(llm_audit, st.session_state.risk_mem, prompt)
        st.session_state.risk_msgs.append({"role": "user", "content": prompt})
        st.session_state.risk_mem.append({"role": "user", "content": prompt})
        if followup_flag == False:
          st.session_state.risk_mem.clear()
          st.session_state.risk_mem.append({"role": "user", "content": prompt})
        history_messages = [ {"role": msg["role"], "content": msg["content"]} for msg in st.session_state.risk_mem]

        
        if followup_flag == True:
            # 5.2.1) Invoke memory_agent with the current thread_id
            config = {"configurable": {"thread_id": st.session_state.session_id}}
            result = memory_agent.invoke({"messages": history_messages}, config=config,)
            rephrased_question = result["messages"][-1].content
        else:
            rephrased_question = prompt
        # 5.2.2) Show what the memory agent decided (for debugging, optional)
        placeholders["Reframed Question with memory"].markdown("## Rephrase Question based on memory")
        placeholders["Reframed Question with memory"].write(rephrased_question)
        

        # ──────────────────────────────────────────────────────────
        # 5.3) Step 2: Call your existing risk‐query pipeline
        # ──────────────────────────────────────────────────────────
        conv, result_df, sql = process_risk_query(llm_audit, rephrased_question)
        #conv, result_df, sql = "Ans",None,"


        
        if conv is None:
            st.chat_message("assistant").write("Sorry, I couldn't answer your question.")
            st.session_state.risk_msgs.append({"role": "assistant", "content": "Sorry, I couldn't answer your question."} )
        else:
            # Show the actual assistant’s final “conversational” response (conv)
            tab1, tab2 = st.tabs(["Conversational", "Tabular"])
            tab1.chat_message("assistant").write(conv)
            tab2.dataframe(result_df, width=600, height=300)
            st.session_state.risk_msgs.append({"role": "assistant", "content": conv})

        end_time = time.time()
        response_time = end_time - start_time
        st.write("response_time",response_time)

        # Format the messages into plain text
        formatted_text = ""
        for i, msg in enumerate(history_messages):
            formatted_text += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
        # Write to a temporary file
        file_name = "chat_history_memory.txt"
        with open(file_name, "w") as f:
            f.write(formatted_text)
        
        # Streamlit download button
        with open(file_name, "rb") as f:
            st.download_button(
                label="Download Chat History Memory",
                data=f,
                file_name=file_name,
                mime="text/plain"
            )

        
            # ---- Simplified Feedback ----           
            # 1. Store the last QA in session_state so it's accessible inside the form
            st.session_state["last_prompt"] = prompt
            st.session_state["last_sql"]    = sql
            st.session_state["last_conv"]   = conv
            st.session_state["session_id"] = st.session_state.session_id
            st.session_state["question_id"] =  uuid.uuid4()
            st.session_state["timestamp"] = datetime.now().isoformat()

            # Callback to handle feedback submission
            def submit_feedback():
                entry = {
                    "session_id":   str(st.session_state["session_id"]),
                    "question_id":  str(st.session_state["question_id"]),
                    "timestamp":  str(st.session_state["timestamp"]),
                    "question": st.session_state.last_prompt,
                    "sql_query": "SQL query: "+ st.session_state.last_sql,
                    "conversational_answer": "Ans: "+ st.session_state.last_conv,
                    "rating": (1+st.session_state.feedback_rating) if st.session_state.feedback_rating else 0,
                    "comments": st.session_state.feedback_comment
                }
                if st.session_state.feedback_rating or st.session_state.feedback_comment:
                    log_to_google_sheets(entry)
                    st.success("Feedback recorded. Thank you!")	
            
                # Clear stored Q&A (optional)
                for k in ("last_prompt", "last_sql", "last_conv"):
                    st.session_state.pop(k, None)

            
            feedback_expander = st.expander("Give Feedback", expanded=False)
            with feedback_expander:
                with st.form("feedback_form"):
                    st.subheader("Rate this answer and leave optional comments")
                
                    # Star rating from 1–5
                    rating = st.feedback(options="stars",key="feedback_rating")
                    # Text feedaback
                    comment = st.text_input("Please provide comments for improvement (optional)",key="feedback_comment")
                    submit = st.form_submit_button("Submit Feedback", on_click=submit_feedback)

            if submit == False:
                entry = { "session_id":   str(st.session_state["session_id"]),
                          "question_id":  str(st.session_state["question_id"]),
                          "timestamp":  str(st.session_state["timestamp"]),
                           "question":  prompt,
                           "sql_query": "SQL query: "+ sql,
                           "conversational_answer": "Ans: "+ conv,
                        }
                log_to_google_sheets(entry)
   
          
records = sheet.get_all_records()
# Convert the records to a pandas DataFrame
df = pd.DataFrame(records)
# Convert the DataFrame to CSV format in memory
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()








# Display the download button in the Streamlit sidebar
st.sidebar.markdown("### 📥 Download Chat Log")
if csv_data:
    st.sidebar.download_button(
        label="Download log (CSV)",
        data=csv_data,
        file_name="chat_log.csv",
        mime="text/csv"
    )
else:
    st.sidebar.write("No log file yet.")

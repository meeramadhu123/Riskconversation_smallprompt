import os
import re
import sqlite3
import mysql.connector
import pandas as pd
import warnings
import streamlit as st
warnings.filterwarnings('ignore')


# Import LangChain utilities for embeddings, vector store, documents, and chains.
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain import PromptTemplate
from langchain.chains import LLMChain

# Import your LLM – here we use ChatOllama as in the previous examples.
#from langchain_ollama.chat_models import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from sentence_transformers import SentenceTransformer

############################################################
# 1. Parse Table Descriptions from a Text File
############################################################
def parse_table_descriptions(file_path):
    """
    Parse a description file into a dictionary with metadata for each table.
    
    The file is expected to have a structure like:
        Table: <table_name>
        Description: <table description>
        Columns:
            Column1: <column description>
            Column2: <column description>
        Relation:
            Related_Table1: Foreign_Key1
            Related_Table2: Foreign_Key2
    
    Returns a dictionary where each key is a table name and the value is a dictionary with:
      - "table_description": the table's description.
      - "columns": a dictionary mapping each column name to its description.
      - "relations": a dictionary mapping related table names to their foreign keys.
    """
    metadata = {}
    current_table = None
    inside_columns = False
    inside_relations = False

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("Table:"):
                current_table = stripped.split("Table:", 1)[1].strip()
                metadata[current_table] = {"table_description": "", "columns": {}, "relations": {}}
                inside_columns = False
                inside_relations = False
            elif stripped.startswith("Description:") and current_table is not None:
                metadata[current_table]["table_description"] = stripped.split("Description:", 1)[1].strip()
                inside_columns = False
                inside_relations = False
            elif stripped.startswith("Columns:") and current_table is not None:
                inside_columns = True
                inside_relations = False
            elif stripped.startswith("Relation:") and current_table is not None:
                inside_columns = False
                inside_relations = True
            elif inside_columns and current_table is not None:
                if ":" in stripped:
                    col_name, col_desc = map(str.strip, stripped.split(":", 1))
                    metadata[current_table]["columns"][col_name] = col_desc
            elif inside_relations and current_table is not None:
                if ":" in stripped:
                    related_table, foreign_key = map(str.strip, stripped.split(":", 1))
                    metadata[current_table]["relations"][related_table] = foreign_key
    
    return metadata


############################################################
# 2. Retrieve Table Metadata from MySQL Database
############################################################
def get_metadata_from_mysql(db_config, descriptions_file=None):
    """
    Connect to a MySQL database and retrieve metadata for each table in the specified schema.
    
    Parameters:
      db_config: A dictionary with keys:
         - user
         - password
         - host
         - port
         - database (schema name)
      descriptions_file: Optional file path to a table descriptions text file.
    
    Returns:
      - MySQL connection object.
      - A metadata dictionary for each table containing:
           - "columns": list of column names.
           - "sample_data": first 2 rows as a list of dictionaries.
           - "table_description": description from file (if available).
           - "relations": relation info from file (if available).
           - "column_descriptions": column description info from file (if available).
    """
    # Establish connection using MySQL connector.
    conn = mysql.connector.connect(
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["database"]
    )
    
    metadata = {}
    
    # Parse table descriptions if provided.
    descriptions = {}
    if descriptions_file and os.path.exists(descriptions_file):
        descriptions = parse_table_descriptions(descriptions_file)
    
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]
    
    for table_name in table_names:
        query = f"SELECT * FROM `{table_name}` LIMIT 2"
        try:
            df = pd.read_sql(query, conn)
            # Build metadata entry.
            metadata[table_name] = {
                "columns": list(df.columns),
                "sample_data": df.to_dict(orient="records"),
                "table_description": descriptions.get(table_name, {}).get("table_description", ""),
                "relations": descriptions.get(table_name, {}).get("relations", {}),
                "column_descriptions": descriptions.get(table_name, {}).get("columns", {})
            }
           # print(f"Retrieved metadata for table '{table_name}'.")
        except Exception as e:
            print(f"Error retrieving data from table {table_name}: {e}")
    
    return conn, metadata



############################################################
# 3. Create a Vector Database (FAISS) from Metadata
############################################################
def create_vector_db_from_metadata(metadata):
    """
    Convert the metadata dictionary into a list of Document objects and create a FAISS vector store.
    Each Document's page_content contains a summary of the table (name, description, columns, sample data).
    """
    documents = []
    for table_name, info in metadata.items():
        doc_text = f'Table Name: "{table_name}"\n'
        if info.get("table_description"):
            doc_text += f"Description: {info['table_description']}\n"
        doc_text += "Columns: " + ", ".join(info.get("columns", [])) + "\n"
        doc_text += f"Relations: {info.get('relations')}\n"
        doc_text += f"Sample Data (first 2 rows): {info.get('sample_data')}\n"
        doc_text += f"Column Descriptions: {info.get('column_descriptions')}\n"
        documents.append(Document(page_content=doc_text, metadata={"table_name": table_name}))
    # Use a HuggingFace embedding model
    model_path=r"all-MiniLM-L6-v2"
    #model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    #model.save(model_path)
    embeddings = HuggingFaceEmbeddings(model_name = model_path,model_kwargs={'device': 'cpu'})
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store




############################################################
# 4. Retrieve Top 10 Tables using RAG from the Vector DB
############################################################
def retrieve_top_tables(vector_store, question, k):
    """
    Use the vector store to perform a similarity search and retrieve the top 10 relevant table documents.
    Returns a dictionary of selected table metadata.
    """
    retrieved_docs = vector_store.similarity_search(question, k)
    selected_metadata = {}
    for doc in retrieved_docs:
        table_name = doc.metadata["table_name"]
        selected_metadata[table_name] = doc.page_content  # Or you could return the full metadata if needed.
    return retrieved_docs  # Return Document objects for further processing.



####################################################################################
# 5. Retrieve Top 3 Tables using Question Table names example set from Top 10 tables
####################################################################################
def create_llm_table_retriever(llm, user_query, top_tables, example_df):
    """
    Create an LLMChain for the second-level retriever prompt.
    
    Args:
        llm: A LangChain-compatible LLM object.
        user_query (str): The user's SQL-related question.
        top_tables (list): List of top 10 table names.
        example_df (df): Dataframe with example questions.
        
    Returns:
        str: LLM outputs 3 relevant table names.
    """
    # Load and format examples
    examples = "\n".join([
        f"- Table: {row['Table_names']}\n  Question: {row['Question']}"
        for _, row in example_df.iterrows()
    ])

    # Prepare template
    template_str = """
    You are an intelligent SQL assistant helping to select the most relevant tables for a given user query.
    
    ## User Query:
    {user_query}
    
    ## Top 10 Retrieved Table Names:
    {top_tables}
    
    ## Example Table Name to User Query Mappings:
    {examples}

    ##Example Output:
    "Table A", "Table B", "Table C"
    
    Based on the examples and the top 10 tables, identify which 3 tables are most relevant to the user's query. Please only list the names
    of these 3 most relevant tables only, no additional information is required. Also note that each table name should be in in double quotes.
    """

    retriever_prompt_template = PromptTemplate(
        input_variables=["user_query", "top_tables", "examples"],
        template=template_str.strip())

    llm_chain = LLMChain(prompt=retriever_prompt_template, llm=llm)

    # Format inputs
    input_dict = { "user_query": user_query,
        "top_tables": "\n".join([f"- {t}" for t in top_tables]),
        "examples": examples
    }
    result = llm_chain.run(input_dict)
    return result




#######################################################################
#6. Reframe the question asked the user for better understanding by LLM
#######################################################################
def question_reframer(selected_docs,user_question,llm):
    selected_metadata_str = ""
    for doc in selected_docs:
        # We assume the document's page_content already has the table metadata.
        selected_metadata_str += doc.page_content + "\n\n"
     
    # Function to reformulate questions using llm
    question_prompt = PromptTemplate(template="""You are a data analysis assistant tasked with reformulating a user's question for later SQL query generation. Your goal is to produce a clear, unambiguous question that:
    - Accurately reflects the user's intent.
    - Uses the exact table names and column names as provided in the metadata.
    - Specifies the correct column-to-table relationships without hallucinating.
    - Clearly outlines any necessary joins, filtering, grouping, or ordering, using the correct table aliases.
    
    Below is the detailed metadata for the selected tables:
    {selected_metadata}
    
    Please follow these instructions precisely:
    1. **Understand the User's Intent:** Analyze the user's question carefully to identify what data is needed and how it should be processed.
    2. **Eliminate Ambiguity:** Remove any vague or generic terms. Rephrase the question to precisely state the required data operation.
    3. **Use Exact Names:** Replace any generic terms with the exact column names and table names from the metadata. Do not invent any names.
    4. **Ensure Accurate Mapping:** For every column mentioned, clearly indicate the corresponding table (using correct table aliases) and specify join conditions if multiple tables are needed.
    5. **Include All Necessary Elements:** Ensure that all relevant tables and columns are mentioned so that a correct SQL query can be constructed in the next stage.
    6. **Avoid Hallucination:** Rely solely on the provided metadata without adding extra or assumed information.
    7. **Self-Verification:** Before finalizing the reformulated question, list the available columns for each table (from the metadata) and cross-check that each column used in your answer is present in the correct table.
    8. **Include only required Tables**: Use only required tables for joining and use correct method of joining (in most cases inner join works).
    9. **Count**: When using COUNT(Col_A) in combination with GROUP BY Col_B, it is recommended to use COUNT(DISTINCT Col_A) to avoid duplicate counts, and ensure that only non-null values of Col_B are included in the grouping.
    10. "Show Col_A by Col_B" suggests distribution of Col_A by Col_B i.e COUNT(DISTINCT Col_A) & GROUP BY Col_B.
    11. When using GROUP BY, ensure that only the columns explicitly requested by the user are included in the grouping. Avoid adding unnecessary columns that may alter the intended grouping results.
    12. In case question has terms like Critical, High, Medium, Low rated Risks it usually refers to inherent_risk_rating and/or residual_risk_rating columns, this is relevant to only risks.
    13. Please Replace risk_type column with risk_category1 in SQL query if it is there.
    14. Please do not confuse KRI_MAIN table and its respective columns with key_risk and key_control columns in RISK_UNIVERSE_MAIN and CONTROL_MAIN table respectively.
    15. When user specifically asks for RCM he requires only risk_id, control_id, risk_name, control_name, inherent_risk_rating, residual_risk_rating, control_rating columns from RU_RCM table.
    16. In case of KRI dependencies are risks connected to those KRIs.
    
    
    
    Now, based on the metadata above and the user's question below, generate only the reformulated question:
    Example Template for Reframed Question: 
          Question: "What is maximum sales in 2020?" 
          Reframed Question:"**Formulation Reasoning**:
                            `Table_A`: `sales`,...
                            `Table_B`: `year`,...
                             **Formulation**: Show maximum value of sum of `sales` from `Table_A` where `year` from `Table_B` is '2020'."

    
    Question: {question}
    Reformulated Question:

        """,
        input_variables=["selected_metadata", "question"])
    
    llm_chain = LLMChain(prompt=question_prompt, llm=llm)
    
    try:
        # Use a chat-based prompt template
        response = llm_chain.run({"question": user_question, "selected_metadata":selected_metadata_str})
        return response
    except Exception as e:
        print("Error generating question:", str(e))
        return None




############################################################
# 7. Generate SQL Query Using Metadata for Selected Tables
############################################################
def generate_sql_query_for_retrieved_tables(selected_docs, user_question, example_df, llm):
    """
    Build a prompt using only the metadata for the selected tables (from the retrieved documents)
    and use the LLM to generate a SELECT SQL query.
    """

    filtered_metadata_str = ""
    for doc in selected_docs:
        lines = doc.page_content.split("\n")  # Split content by lines
        extracted_info = {}

        for i, line in enumerate(lines):
            if line.startswith("Table Name:"):
                extracted_info["Table Name"] = line.replace("Table Name: ", "").strip()
            elif line.startswith("Columns:"):
                extracted_info["Columns"] = line.replace("Columns: ", "").strip()
            elif line.startswith("Relations"):
                extracted_info["Relations"] = line.replace("Relations: ", "").strip()
            

        # Format the extracted information into the final string
        filtered_metadata_str += f"Table Name: {extracted_info.get('Table Name', 'N/A')}\n"
        filtered_metadata_str += f"Columns: {extracted_info.get('Columns', 'N/A')}\n"
        filtered_metadata_str += f"Relations: {extracted_info.get('Relations', 'N/A')}\n"
        
      # Load and format examples
    examples = "\n".join([
        f"- Question: {row['Question']}\n  SQL Queries: {row['SQL Queries']}"
        for _, row in example_df.iterrows()
    ])
    
    
    sql_prompt_template = PromptTemplate(template="""
                You are a data assistant with access to a MySQL database containing a subset of tables.  
        ## Below is the metadata for the selected tables:  
        
        {selected_metadata}  

        ## Below are few example sets of questions and their respective SQL queries for your reference:

        {Question_SQL_Queries_Examples} 
        
        Strictly generate only a SELECT SQL query to answer the user's question while following these instructions:  
        
        1. **Output Only the SQL Query** – Do not include any explanations or additional text. Don't add sql word before or after the query. 
        2. **Valid SQL Syntax** – Strictly ensure that the generated query is syntactically correct for MySQL. Very important task.   
        3. **Proper Table and Column Naming** – Use the exact table and column names as provided in the metadata.  
        4. **Safe Query Structure** – Avoid SQL errors by:  
           - Enclosing column names in **backticks** (`"`).
           - Enclosing table names in **backticks** (`"`).
           - Enclosing table aliases in **backticks** (`"`).
           - Please never use tablenames or aliases within double quotes.
           - Using **explicit JOINs** where necessary.  
           - Using **LIKE '%'** instead of unknown string values in filters.  
        5. **Consistent Naming & Uniqueness** – Ensure queries focus only on unique:  
           - Risks, Controls, Issues, Actions, Risk Registers, Causes, Impacts, Mitigation Plans, Risk Programs, and Risk Program Schedules, even if not explicitly mentioned.  
        6. **Clear Aliasing** – Assign meaningful aliases to tables. This is important. 
        7. **Column Ambiguity Elimination** – Avoid ambiguous column references by **always specifying the table alias** when selecting or filtering columns. This is very important.
        8. **Use Aggregation or DISTINCT as Needed** – If a column might contain duplicates, ensure the query retrieves only unique records using `DISTINCT`.  
        7. **Ensure Correct Joins** –  
           - Use appropriate **INNER JOIN, LEFT JOIN, or RIGHT JOIN** when multiple tables are involved.  
           - Always specify the **correct primary and foreign key relationships** from the metadata. 
        8. **Very Important**- In query use the tablename where all informations/columns relevant ot user question is available, in case it is not available, please join with other tablename where this information is available.
        9. **Fuzzy matching **-Please use LIKE % incase of fuzzy matching with string for filtering purpose when there is a doubt about actual value/condition. Please prefer LIKE instead of '=' wherever applicable in SQL query.
        10. **Column name**- Show (SELECT) all the required column names/counts/aggregates(sum,max,min,avg etc.) from required table/tables to answer the question correctly.
        11. Please Replace risk_type column with risk_category1 in SQL query if it is there.
        12. In case question has terms like Critical, High, Medium, Low rated Risks it usually refers to inherent_risk_rating and/or residual_risk_rating columns, this is relevant to only risks.
        13. Please do not confuse KRI_MAIN table and its respective columns with key_risk and key_control columns in RISK_UNIVERSE_MAIN and CONTROL_MAIN table respectively.
        14. **Count**: When using COUNT(Col_A) in combination with GROUP BY Col_B, it is recommended to use COUNT(DISTINCT Col_A) to avoid duplicate counts, and ensure that only non-null values of Col_B are included in the grouping.
        15. "Show Col_A by Col_B" suggests distribution of Col_A by Col_B i.e COUNT(DISTINCT Col_A) & GROUP BY Col_B.
        
        ## User's Question: {question}  
        ## SQL Query: 
        """,input_variables=["selected_metadata","Question_SQL_Queries_Examples", "question"])

    llm_chain_sql = LLMChain(prompt=sql_prompt_template, llm=llm)
    sql_query = llm_chain_sql.run({
        "selected_metadata": filtered_metadata_str,
        "Question_SQL_Queries_Examples": examples,
        "question": user_question
    })
    return sanitize_query(sql_query)


############################################################
# 8. Helper Functions: Sanitize Query, Execute Query, Analyze Result
############################################################
def sanitize_query(query):
    #query = re.sub(r'<think>.*?</think>', '', query, flags=re.DOTALL | re.IGNORECASE)
    # Check if the word "SQL" is present
    if "SQL" in query.upper():
        # Remove all instances of the word "SQL" (case insensitive)
        query = query.replace("SQL", "").replace("sql", "")
        query = query.replace("```", "")
        return query
    else:
        return query

def execute_sql_query(conn, query):
    try:
        error_msg = ''
        #cursor = conn.cursor()
        #cursor.execute(query)
        #cols = [desc[0] for desc in cursor.description] if cursor.description else []
        #rows = cursor.fetchall()
        #cursor.close()
        #return pd.DataFrame(rows, columns=cols), error_msg
        df=pd.read_sql(query,conn)
        return df, error_msg
    except Exception as e:
        print("Error executing SQL query:", str(e))
        return None, str(e)


def analyze_sql_query(user_question, tabular_answer, llm):
    template_prompt = PromptTemplate(template="""
        You are an experinced data analyst specialised in risk analytics domain. Below is the user's question:
        Question: {question}
        
        And here is the answer given in tabular format obtained by running an SQL query:
        Tabular Answer: {tabular_answer}

        1. Please provide accurate and relevant answer to users question.
        2. Provide a conversational answer as concise analysis or summary of the results in bullet points or a short sentence.
        3. Please do not hallucinate and be specific with answers.
        4. Please don't ask users any addtional questions but only provide accurate answer.
        
         
        """, input_variables=["question", "tabular_answer"])
    try:
        llm_conv_chain = LLMChain(prompt=template_prompt, llm=llm)
        response = llm_conv_chain.run({"question": user_question, "tabular_answer": tabular_answer})
        return response.strip()
    except Exception as e:
        return "Sorry, I was not able to answer your question"

def finetune_conv_answer(user_question, conv_result, llm):
    template_prompt = PromptTemplate(template="""
        You are an experienced Risk Management Analyst specialised in risk analytics domain. Below is the user's question:
        Question: {question}
        
        And here is the answer given by LLM model:
        Answer: {conv_answer}

        Rephrase the answer generated by LLM in a professional way based on following requirements:
        1. Please provide accurate and relevant answer to users question.
        2. Provide answer in bullet points or a short sentence.
        3. Please do not hallucinate and be specific with answers.
        4. Please don't ask users any addtional questions but only provide accurate answer.
        5. Very Important: Please remove any redundant/unwanted sentence or bullet points from the answer.
        
         
        """, input_variables=["question", "conv_answer"])
    try:
        llm_conv_chain = LLMChain(prompt=template_prompt, llm=llm)
        response = llm_conv_chain.run({"question": user_question, "conv_answer": conv_result})
        return response.strip()
    except Exception as e:
        return "Sorry, I was not able to answer your question"



def debug_query(selected_docs, user_question, sql_query, llm, error):
    """
    Build a prompt using  the metadata for the selected tables (from the retrieved documents), error message
    and use the LLM to correct the SELECT SQL query.
    """

    filtered_metadata_str = ""
    for doc in selected_docs:
        lines = doc.page_content.split("\n")  # Split content by lines
        extracted_info = {}

        for i, line in enumerate(lines):
            if line.startswith("Table Name:"):
                extracted_info["Table Name"] = line.replace("Table Name: ", "").strip()
            elif line.startswith("Columns:"):
                extracted_info["Columns"] = line.replace("Columns: ", "").strip()
            elif line.startswith("Relations"):
                extracted_info["Relations"] = line.replace("Relations: ", "").strip()
            elif line.startswith("Sample Data (first 2 rows):"):
                extracted_info["Sample Data"] = line.replace("Sample Data (first 2 rows): ", "").strip()

        # Format the extracted information into the final string
        filtered_metadata_str += f"Table Name: {extracted_info.get('Table Name', 'N/A')}\n"
        filtered_metadata_str += f"Columns: {extracted_info.get('Columns', 'N/A')}\n"
        filtered_metadata_str += f"Relations: {extracted_info.get('Relations', 'N/A')}\n"
        filtered_metadata_str += f"Sample Data (first 2 rows): {extracted_info.get('Sample Data', 'N/A')}\n\n"

    sql_prompt_template = PromptTemplate(template="""
        You are a data assistant with access to a MySQL database containing a subset of tables. Below is the metadata for the selected tables:
        {selected_metadata}
        
        A SQL query was generated to answer the user's question, but it produced the following error:
        Error: {error}
        
        User's Question: {question}
        Original SQL Query: {sql_query}
        
        Your task is to debug and rewrite the SQL query carefully. Follow these strict instructions:
        
        1. **Output Only the SQL Query:** Do not include any explanations or extra text.
        2. **Valid MySQL Syntax:** Ensure that the rewritten query is syntactically correct for MySQL.
        3. **Proper Naming with Backticks:** Use the exact table names, column names, and aliases as provided in the metadata. Enclose them in backticks (`) and never in double quotes.
        4. **Correct Join Conditions:** When multiple tables are involved, use explicit JOINs with the correct primary and foreign key relationships as indicated in the metadata.
        5. **Eliminate Ambiguities:** Always qualify column references with their table alias to avoid ambiguity.
        6. **Fix the Error:** Analyze the provided error message and adjust the query accordingly.
        7. **Accurate and Complete:** Ensure the query retrieves all necessary columns or aggregates to correctly answer the user's question.
        8. **Fuzzy Matching:** Use LIKE with '%' for string filters if needed.
        
        Based on the above, rewrite the SQL query to correctly answer the user's question.
        
        SQL Query:
 
        """,input_variables=["selected_metadata","error", "question","sql_query"])

    llm_chain_sql = LLMChain(prompt=sql_prompt_template, llm=llm)
    sql_query = llm_chain_sql.run({
        "selected_metadata": filtered_metadata_str,
        "error": error,
        "question": user_question,
        "sql_query": sql_query
    })
    return sanitize_query(sql_query)


############################################################
# 9. Main Chatbot Function: Orchestrate the SQL Agent with RAG
############################################################
def run_chatbot(llm, descriptions_file, examples_file, db_config):
    """
    Build a SQLite database from CSV files (and attach table/column descriptions if provided),
    convert metadata into a vector DB, use retrieval (RAG) to select the most relevant tables,
    generate an SQL query using the metadata for those tables, execute the query, and display
    the results along with a conversational analysis.
    """
    # 1. Build the SQLite DB and load metadata.
    conn, metadata = get_metadata_from_mysql(db_config, descriptions_file=descriptions_file)
    if conn is None or not metadata:
        print("Database initialization failed.")
        return

    #print(metadata)
    # 2. Convert metadata to a vector database.
    vector_store = create_vector_db_from_metadata(metadata)
    while True:
        user_question = input("\nAsk a question about the database (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            break

        # 3. Retrieve the top 10 relevant table metadata documents using RAG.
        retrieved_docs = retrieve_top_tables(vector_store, user_question, k=10)
        top_table_names=[]
        example_df=pd.read_excel(examples_file)
        for doc in retrieved_docs:
            table=doc.metadata["table_name"]
            top_table_names.append(table)
        top_3_tables=create_llm_table_retriever(llm,user_question, top_table_names, example_df)
        filtered_docs = [doc for doc in retrieved_docs if doc.metadata["table_name"] in top_3_tables]
        # 4. Reframe the question asked by user
        reframed_question = question_reframer(filtered_docs,user_question,llm)
        # 5. Generate SQL query based on the retrieved metadata.
        sql_query = generate_sql_query_for_retrieved_tables(filtered_docs, reframed_question, example_df, llm)
        print("SQL Query Generated","\n", sql_query)
        # 6. Execute the SQL query and display results.
        result,error = execute_sql_query(conn, sql_query)
        if result is not None and not result.empty:
            print("\nTabular Result:")
            print(result)
            conv_result = analyze_sql_query(user_question, result.to_dict(orient="records"), llm)
            print("\nConversational Analysis:")
            conv_result= finetune_conv_answer(user_question, conv_result, llm)
            print(conv_result)
        else:
            #print("Error:",error)
            corr_sql_query = debug_query(retrieved_docs,user_question, sql_query,llm, error)
            print("Debugged SQL Query Generated","\n", corr_sql_query)
            result,error = execute_sql_query(conn, corr_sql_query)
            if result is not None and not result.empty:
                print("\nTabular Result:")
                print(result)
                conv_result = analyze_sql_query(user_question, result.to_dict(orient="records"), llm)
                print("\nConversational Analysis:")
                conv_result= finetune_conv_answer(user_question, conv_result, llm)
                print(conv_result)
            else:
                print("Failed to exceute the quey after trying twice due to the following Error:",error) 

if __name__ == "__main__":
    # Initialize your LLM (here using ChatOllama; adjust parameters as needed)
    #llm = ChatOllama(model='qwen2.5:14b', temperature=0, num_ctx=50000)

    # Set the path to your table descriptions text file.
    descriptions_file = "/kaggle/input/rag-data/all_table_metadata.txt" # UPDATE THIS PATH
    # Set the path to your examples excel file.
    examples_file = '/kaggle/input/rag-data/Example question datasets.xlsx'
    #MySQL DB Connection details
    db_config = {
        "user": DATABASE_USER,
        "password": DATABASE_PASSWORD,
        "host": DATABASE_HOST,
        "port": DATABASE_PORT,
        "database": Schema_Name
    }
    llm = ChatNVIDIA(model="qwen/qwen2.5-coder-32b-instruct",
                     api_key="",
                    temperature=0, num_ctx=50000)


    
    # Start the SQL Agent chatbot using RAG.
    run_chatbot(llm, descriptions_file,examples_file, db_config)

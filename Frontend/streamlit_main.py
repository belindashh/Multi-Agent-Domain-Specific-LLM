import streamlit as st
import requests
from grobid_client.grobid_client import GrobidClient
import os
import time
import json
from Frontend.utils import *

API_URL = "http://127.0.0.1:8000/api/"
UPDATE_FILE_URL = API_URL + "update_file"
GENERAL_GPT_URL = API_URL + "get_general_gpt"
MATH_GPT_URL = API_URL + "get_math_gpt"
FIND_FILE_GPT_URL = API_URL + "get_file_gpt"
FIND_INFO_GPT_URL = API_URL + "get_local_info_gpt"
TABLE_GPT_URL = API_URL + "get_table_gpt"
MAIN_GPT = API_URL + "get_main_gpt"

def is_latex(chunk):
    latex_keywords = ["\\frac", "_", "^", "\\cdot", "\\text", "\\times", "\\sum", "\\int"]
    return any(symbol in chunk for symbol in latex_keywords) or "$" in chunk


def check_payload(query):
    payload = query
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(MAIN_GPT, json=payload, headers=headers)
        return response.json()
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to FastAPI server. Make sure it is running."
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    
def get_gpt_response(query):
    payload = {"query": query}
    headers = {'Content-Type': 'application/json'}

    try:
        if check_payload(payload).get("response") == "general":
            response = requests.post(GENERAL_GPT_URL, json=payload, headers=headers)
        elif check_payload(payload).get("response")  == "calculations":
            response = requests.post(MATH_GPT_URL, json=payload, headers=headers)
        elif check_payload(payload).get("response")  == "engineering_file_search":
            response = requests.post(FIND_FILE_GPT_URL, json=payload, headers=headers)
        elif check_payload(payload).get("response")  == "engineering_info_search":
            response = requests.post(FIND_INFO_GPT_URL, json=payload, headers=headers)
        elif check_payload(payload).get("response")  == "tables":
            response = requests.post(TABLE_GPT_URL, json=payload, headers=headers)
        else:
            return "error"
        return (response.json()).get("response")
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to FastAPI server. Make sure it is running."
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    
def fetch_chat_history(selected_db):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute(f"SELECT user_message, assistant_message FROM {selected_db}")
    messages = cursor.fetchall()
    cursor.close()
    connection.close()
    return messages
    
def export_chat_history(selected_db):
    if selected_db:
        chat_data = fetch_chat_history(selected_db)
        df = pd.DataFrame(chat_data)
        csv_data = df.to_csv(index=False)
        st.download_button(
            label=f"📥 Export '{selected_db}' Chat History as CSV",
            data=csv_data,
            file_name=f"{selected_db}_chat_history.csv",
            mime="text/csv"
        )

def main():
    st.title("Chatbot & File Upload")


    tab1, tab2 = st.tabs(["Chatbot", "Upload PDF"])
    
    available_databases = ["chat_history", "math_chat_history", "searchinfo_chat_history", "search_chat_history", "table_chat_history"]

    with tab1:
        st.markdown("""
            <style>
                .stTextInput, .stTextArea {
                    position: fixed;
                    bottom: 20px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 60%;
                    z-index: 100;
                }
            </style>
        """, unsafe_allow_html=True)

        st.subheader("Chat with AI")

        selected_db = st.selectbox("Select a chat database:", available_databases)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        input_placeholder = st.empty()
        prompt = input_placeholder.chat_input("What is up?")

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            response_text = ""
            response = get_gpt_response(prompt)
            response = response.replace("\\[\n", "\\[")
            response = response.replace("\n\\]", "\\]")
            response_chunk = response.splitlines()
            with st.chat_message("assistant"):
                for line in response_chunk:
                    if "\\["in line:
                        line = line.replace("\\[", "$$")
                    if "\\]" in line:
                        line = line.replace("\\]", "$$")
                    if "\\("in line:
                        line = line.replace("\\(", "$$")
                    if "\\)" in line:
                        line = line.replace("\\)", "$$")
                    st.markdown(line)
                    response_text += f"\n{line}"

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()

        export_chat_history(selected_db)
    
    with tab2:
        st.subheader("Upload & Process PDF")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        if uploaded_file is not None:
            temp_path = f"./temp_files/temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            client = GrobidClient(config_path="config.json", timeout=300)
            client.process("processFulltextDocument", "temp_files", output="./temp_files/", consolidate_citations=True , tei_coordinates=True, force=True)
            tei_file = temp_path.replace(".pdf", ".grobid.tei.xml")
            counter=0
            while not os.path.exists(tei_file) and counter != 300:
                print("⏳ Waiting for GROBID to process the document...", end="\r", flush=True)
                time.sleep(10)
                counter += 10
            try:
                payload = {"query": tei_file, 
                           "file_name": uploaded_file.name}
                headers = {'Content-Type': 'application/json'}
                st.subheader("Extracting text using GROBID...")
                response = requests.put(UPDATE_FILE_URL, json=payload, headers=headers)
                time.sleep(30)
                st.success(f"File uploaded: {uploaded_file.name}")
                os.remove(temp_path)
                if os.path.exists(tei_file):
                    os.remove(tei_file)
            except:
                st.error(f"File upload failed: {uploaded_file.name}")


if __name__ == "__main__":
    main()

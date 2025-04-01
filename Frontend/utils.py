import string
from openai import OpenAI
from pydantic import BaseModel
import mysql.connector 
from Frontend.config import Config
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import tiktoken
from scipy import spatial
import ast
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import nltk
from collections import Counter
nltk.download('punkt')
nltk.download('stopwords')
import json
import pandas as pd 

set(stopwords.words('english'))

DB_name = Config.DB_NAME 
Chat_TB = Config.CHAT_TB_NAME
Data_TB = Config.DATA_TB_NAME
Math_TB = Config.MATH_TB_NAME
Table_TB = Config.TABLE_TB_NAME
Search_TB = Config.SEARCH_TB_NAME
SearchInfo_TB = Config.SEARCHINFO_TB_NAME

GPT_MODEL_4_MINI = Config.OPENAI_MODEL
client = OpenAI(api_key= Config.OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-ada-002"

def remove_punctuation(text):
    return str(text).translate(str.maketrans('', '', string.punctuation))

def get_db_connection():
    conn = mysql.connector.connect(user='root',  
                               password='',  
                               host='localhost',  
                               database=DB_name) 
    return conn

def generate_embedding(text: str) -> list[float]:
	resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text,)

	return resp.data[0].embedding

def get_chat_history(type:str=""):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    if type == "":
        cursor.execute(f"SELECT user_message, assistant_message FROM {Chat_TB} ORDER BY timestamp DESC LIMIT 10")
    elif type == "math":
        cursor.execute(f"SELECT user_message, assistant_message FROM {Math_TB} ORDER BY timestamp DESC LIMIT 10")
    elif type == "local_file":
        cursor.execute(f"SELECT user_message, assistant_message FROM {Search_TB} ORDER BY timestamp DESC LIMIT 10")
    elif type == "local_info":
        cursor.execute(f"SELECT user_message, assistant_message FROM {Search_TB} ORDER BY timestamp DESC LIMIT 10")
    elif type == "table":
        cursor.execute(f"SELECT user_message, assistant_message FROM {Table_TB} ORDER BY timestamp DESC LIMIT 10")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    chat_history = []
    for row in rows:
        if row['user_message']:
            user_text = row['user_message']
            chat_history.append({"role": "user", "content": user_text})
        if row['assistant_message']:
            asst_text = row['assistant_message']
            chat_history.append({"role": "assistant", "content": asst_text})
    
    return chat_history[::-1]

def save_chat_history(user_message, assistant_message, type):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"INSERT INTO {Chat_TB} (user_message, assistant_message) VALUES (%s, %s)", 
                   (user_message, assistant_message))
    if type == "table":
        cursor.execute(f"INSERT INTO {Table_TB} (user_message, assistant_message) VALUES (%s, %s)", 
                   (user_message, assistant_message))
    elif type == "math":
        cursor.execute(f"INSERT INTO {Math_TB} (user_message, assistant_message) VALUES (%s, %s)", 
                   (user_message, assistant_message))
    elif type == "local_file":
        cursor.execute(f"INSERT INTO {Search_TB} (user_message, assistant_message) VALUES (%s, %s)", 
                   (user_message, assistant_message))
    elif type == "local_info":
        cursor.execute(f"INSERT INTO {SearchInfo_TB} (user_message, assistant_message) VALUES (%s, %s)", 
                   (user_message, assistant_message))
    conn.commit()
    cursor.close()
    conn.close()

def main_gpt(query: str, model: str = GPT_MODEL_4_MINI, token_budget: int = 4096 - 500, print_message: bool = False) -> str:
    messages = [{"role": "system", "content": f'Return only the category in one word. Classify the following user input into one of five categories: "calculations", "engineering_file_search", "engineering_info_search", "tables" or "general". User Input: "{query}"'}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = (response.choices[0].message.content).strip().lower()
    print(response_message)

    if print_message:
        print(f"Assistant: {response_message}")

    return response_message

def general_gpt(query: str, model: str = GPT_MODEL_4_MINI, token_budget: int = 4096 - 500, print_message: bool = False) -> str:
    chat_history = get_chat_history()

    chat_history.append({"role": "user", "content": query})

    messages = [{"role": "system", "content": "respond to user's questions"}] + chat_history

    while len(messages) > token_budget:
        messages.pop(0) 

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    print(response_message)
    chat_history.append({"role": "assistant", "content": response_message})

    if print_message:
        print(f"Assistant: {response_message}")

    return response_message

def math_gpt(query: str, model: str = GPT_MODEL_4_MINI, token_budget: int = 4096 - 500, print_message: bool = False) -> str:
    chat_history = get_chat_history("math")

    chat_history.append({"role": "user", "content": query})

    messages = [{"role": "system", "content": "You are a scientist that is very good  at math and science calculations. Analyse and provide answers to user's question"}] + chat_history

    while len(messages) > token_budget:
        messages.pop(0) 

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    print(response_message)

    chat_history = get_chat_history()
    chat_history.append({"role": "assistant", "content": response_message})

    chat_history = get_chat_history("math")
    chat_history.append({"role": "assistant", "content": response_message})

    if print_message:
        print(f"Assistant: {response_message}")

    return response_message

def table_gpt(query: str, model: str = GPT_MODEL_4_MINI, token_budget: int = 4096 - 500, print_message: bool = False) -> str:
    chat_history = get_chat_history("table")

    chat_history.append({"role": "user", "content": query})

    messages = [{"role": "system", "content": "You are a researcher who is able to analyse research papers and summarise them into proper tables. Create a table based on the column descriptions given and the relevant information provided by user. Do not fill in any information that you are unsure of."}] + [{"role": "user", "content": query}]

    while len(messages) > token_budget:
        messages.pop(0) 

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    print(response_message)

    chat_history = get_chat_history()
    chat_history.append({"role": "assistant", "content": response_message})

    chat_history = get_chat_history("table")
    chat_history.append({"role": "assistant", "content": response_message})

    if print_message:
        print(f"Assistant: {response_message}")

    return response_message

def add_to_table_gpt(query: str, model: str = GPT_MODEL_4_MINI, token_budget: int = 4096 - 500, print_message: bool = False) -> str:
    chat_history = get_chat_history("table")

    chat_history.append({"role": "user", "content": query})

    messages = [{"role": "system", "content": "You are a trained researcher. If there is data missing from the table given by user, fill in missing information from credible sources to the best of your abilities. Respond with the finalised table and sources of where the missing information is from. Do not fill in with hypothetical values."}] + [{"role": "user", "content": query}]

    while len(messages) > token_budget:
        messages.pop(0) 

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    print(response_message)

    chat_history = get_chat_history()
    chat_history.append({"role": "assistant", "content": response_message})

    chat_history = get_chat_history("table")
    chat_history.append({"role": "assistant", "content": response_message})

    if print_message:
        print(f"Assistant: {response_message}")

    return response_message

def strings_ranked_by_relatedness(query: str, df: pd.DataFrame, column:str, relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y), top_n: int = 100,) -> tuple[list[str], list[float]]:
    query_embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=query,)
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (
            row[column], 
            relatedness_fn(
                query_embedding, 
                np.array(ast.literal_eval(row["content_bigram_embed"])) 
            )
        )
    for _, row in df.iterrows()
]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL_4_MINI) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(query: str,df: pd.DataFrame,model: str,token_budget: int, column: str) -> str:
    strings, relatednesses = strings_ranked_by_relatedness(query, df, column)
    introduction = 'Use the following information to summarise for the Question. Return all relevant information. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    files_included = []
    for idx, string in enumerate(strings):
        file_name = df.iloc[idx]['file_name']
        files_included.append(file_name)
        next_article = string
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article

    message += f"\n\nFiles included: {', '.join(files_included)}"
    return message + question

def full_compilation(query: str,model: str = GPT_MODEL_4_MINI,token_budget: int = 4096 - 500,print_message: bool = False,) -> str:
    conn = get_db_connection()
    call = f"SELECT file_name, content, content_bigram_embed FROM {Data_TB}"
    df = pd.read_sql(call, conn)
    message = query_message(query, df, model=model, token_budget=token_budget, column="content")
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": '''You are a researcher, give the user all information relevant to the user's query. Respond only with paragraphs and do not respond in table format.'''},
        {"role": "user", "content": '''What is the relevant information, and include the file names these information is sourced from:
        
            <info>''' + message + '''</info> ?'''},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    chat_history = get_chat_history()
    chat_history.append({"role": "assistant", "content": response_message})

    chat_history = get_chat_history("local_info")
    chat_history.append({"role": "assistant", "content": response_message})

    if print_message:
        print(f"Assistant: {response_message}")

    return response_message

def summarise_gpt(query: str,model: str = GPT_MODEL_4_MINI,token_budget: int = 4096 - 500,print_message: bool = False,) -> str:
    conn = get_db_connection()
    call = f"SELECT file_name, content, content_bigram_embed FROM {Data_TB}"
    df = pd.read_sql(call, conn)
    message = query_message(query, df, model=model, token_budget=token_budget, column="content")
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": '''You are a researcher, compile and organise a comprehensive summary according to user's query'''},
        {"role": "user", "content": '''What is the summary for the following information, and include the file names these information is sourced from:
        
            <info>''' + message + '''</info> ?'''},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    chat_history = get_chat_history()
    chat_history.append({"role": "assistant", "content": response_message})

    chat_history = get_chat_history("local_info")
    chat_history.append({"role": "assistant", "content": response_message})

    if print_message:
        print(f"Assistant: {response_message}")

    return response_message


def summarise_file_gpt(query: str,model: str = GPT_MODEL_4_MINI,token_budget: int = 4096 - 500,print_message: bool = False,) -> str:
    conn = get_db_connection()
    call = f"SELECT file_name, content, content_bigram_embed FROM {Data_TB}"
    df = pd.read_sql(call, conn)
    message = query_message(query, df, model=model, token_budget=token_budget, column="file_name")
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": '''You are a researcher, organise the relevant file names based on the information provided'''},
        {"role": "user", "content": '''What is the summary for the following information:
        
            <info>''' + message + '''</info> ?'''},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    chat_history = get_chat_history()
    chat_history.append({"role": "assistant", "content": response_message})

    chat_history = get_chat_history("local_info")
    chat_history.append({"role": "assistant", "content": response_message})

    if print_message:
        print(f"Assistant: {response_message}")

    return response_message

def ngram(text, n):
    list_ngrams = []
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text) 
    
    filtered_sentence = [] 
  
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    ngram_text = ngrams(text.split(), n)
    for grams in ngram_text:
        list_ngrams.append(grams[0])

    return list_ngrams

def filter_ngrams(ngram_list, remove_count=1):
    ngram_freq = Counter(ngram_list)
    sorted_ngrams = ngram_freq.most_common()
    filtered_ngrams = sorted_ngrams[remove_count:]
    return [ngram for ngram, _ in filtered_ngrams]

def chunk_text(text, max_tokens=5000, model="text-embedding-ada-002"):

    encoding = tiktoken.encoding_for_model(model) 
    words = text.split() 
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        word_token_count = len(encoding.encode(word))

        if current_tokens + word_token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(word)
        current_tokens += word_token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def split_list_by_token_limit(text_list, max_tokens=5000, model="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model)
    
    chunks = []
    current_chunk = []
    current_tokens = 0

    for text in text_list:
        token_count = len(encoding.encode(text))  

        if token_count > max_tokens:
            sub_chunks = [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]
            for sub in sub_chunks:
                if current_tokens + len(encoding.encode(sub)) > max_tokens:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                current_chunk.append(sub)
                current_tokens += len(encoding.encode(sub))
            continue 

        if current_tokens + token_count > max_tokens:
            chunks.append(" ".join(current_chunk)) 
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append(text)
        current_tokens += token_count
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def add_file2database(file_name: str, desc_text: str, title_text: str, abstract_text: str, body_text: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT title FROM data WHERE file_name LIKE %s"
    cursor.execute(query, (f"%{file_name}%",))
    results = cursor.fetchall()
    if results:
        print("Existing File, don't require upload")
    else:
        abstract_embed = generate_embedding(abstract_text)
        abstract_bigrams = filter_ngrams(ngram(title_text+abstract_text,2), remove_count=1)
        abstract_bigram_embed = generate_embedding(str(abstract_bigrams))
        chunk_body= chunk_text(body_text)
        for i, chunk in enumerate(chunk_body):
            chunk_content = chunk
            chunk_embed = generate_embedding(str(chunk))

            combined_text = title_text + " " + abstract_text + " " + chunk_content
            list_bigrams = filter_ngrams(ngram(combined_text,2), remove_count=1)
            chunks = split_list_by_token_limit(list_bigrams)

            for i, chunk in enumerate(chunks):
                insert_query = """
                INSERT INTO data (file_name, title, description, abstract, abstract_embed, abstract_bigrams, abstract_bigram_embed, content, content_embed, content_bigrams, content_bigram_embed)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    str(file_name),
                    title_text,
                    desc_text,
                    abstract_text,
                    json.dumps(abstract_embed), 
                    str(abstract_bigrams),
                    json.dumps(abstract_bigram_embed),
                    chunk_content,
                    json.dumps(chunk_embed),
                    str(chunk),
                    json.dumps(generate_embedding(str(chunk))),
                )

        cursor.execute(insert_query, values)
        conn.commit()

        cursor.close()
        conn.close()
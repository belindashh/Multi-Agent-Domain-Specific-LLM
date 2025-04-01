from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from Frontend.utils import *
from bs4 import BeautifulSoup
load_dotenv()

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Default Page"}

class QueryRequest(BaseModel):
    query: str

class UploadFile(BaseModel):
    query: str
    file_name: str
    

@app.post("/api/get_main_gpt")
async def api_maingpt(request: QueryRequest):
    query = request.query

    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    response_message = main_gpt(query)

    return {
        "query": query,
        "response": response_message
    }

@app.post("/api/get_general_gpt")
async def api_generalgpt(request: QueryRequest):
    query = request.query

    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    response_message = general_gpt(query)
    save_chat_history(query, response_message, "general")

    return {
        "query": query,
        "response": response_message
    }

@app.post("/api/get_math_gpt")
async def api_generalgpt(request: QueryRequest):
    query = request.query

    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    response_message = math_gpt(query)
    save_chat_history(query, response_message, "math")

    return {
        "query": query,
        "response": response_message
    }

@app.post("/api/get_file_gpt")
async def api_gpt(request: QueryRequest):
    query = request.query

    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    response_message = summarise_file_gpt(query)
    save_chat_history(query, response_message, "local_file")
    

    return {
        "query": query,
        "response": response_message
    }


@app.post("/api/get_local_info_gpt")
async def api_gpt(request: QueryRequest):
    query = request.query

    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    response_message = summarise_gpt(query)
    save_chat_history(query, response_message, "local_info")
    

    return {
        "query": query,
        "response": response_message
    }

@app.post("/api/get_table_gpt")
async def api_gpt(request: QueryRequest):
    query = request.query

    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    gather_info = full_compilation(query)
    print(gather_info)
    pre_response_message = table_gpt(query + "\n Relevant Information:" + gather_info)
    response_message = add_to_table_gpt(pre_response_message)
    save_chat_history(query, response_message, "table")
    

    return {
        "query": query,
        "response": response_message
    }


@app.put("/api/update_file")
async def update_file2sql(request: UploadFile): 
    tei_file = request.query
    main_file = request.file_name
    with open(tei_file, "r", encoding="utf-8") as f:
        tei_content = f.read()

    soup = BeautifulSoup(tei_content, "xml")
    for ref in soup.find_all("ref"):
        ref.unwrap()
    desc_text = soup.fileDesc.get_text(separator="\n", strip=True)
    title_text = soup.title.get_text(separator="\n", strip=True)
    abstract_text = soup.abstract.get_text(separator="\n", strip=True)
    body_text = soup.body.get_text(separator="\n", strip=True)
    add_file2database(main_file, desc_text, title_text, abstract_text, body_text)

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from pydantic import BaseModel
import bcrypt
import json
from database import add_user, find_user, save_analysis, get_last_analysis, get_all_analyses
from model import identify_patient

app = FastAPI()

class UserRegister(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class AnalyzeRequest(BaseModel):
    email: str
    data: list

@app.post("/register")
def register(user: UserRegister):
    if find_user(user.email):
        raise HTTPException(status_code=400, detail="User already exists")
    
    hashed_password = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt()).decode()
    add_user(user.email, hashed_password)
    return {"message": "User registered successfully"}

@app.post("/login")
def login(user: UserLogin):
    db_user = find_user(user.email)
    if not db_user or not bcrypt.checkpw(user.password.encode(), db_user["password"].encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful"}

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    try:
        result = identify_patient(request.data)
        save_analysis(request.email, request.data, result)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/json")
async def analyze_json(email: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = json.loads(contents)
        result = identify_patient(data)
        save_analysis(email, data, result)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/profile/{email}")
def get_profile(email: str):
    last_analysis = get_last_analysis(email)
    if not last_analysis:
        raise HTTPException(status_code=404, detail="No analysis found")
    return {"last_analysis": last_analysis}

@app.get("/history/{email}")
def get_history(email: str):
    analyses = get_all_analyses(email)
    if not analyses:
        raise HTTPException(status_code=404, detail="No history found")
    return {"history": analyses}
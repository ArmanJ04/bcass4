from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["diplom"]

users_collection = db["users"]
results_collection = db["results"]

def add_user(email, password):
    users_collection.insert_one({"email": email, "password": password})

def find_user(email):
    return users_collection.find_one({"email": email})

def save_analysis(email, data, result):
    results_collection.insert_one({
        "email": email,
        "data": data,
        "result": result,
        "timestamp": datetime.now()
    })

def get_last_analysis(email):
    return results_collection.find_one({"email": email}, sort=[("_id", -1)])

def get_all_analyses(email):
    return list(results_collection.find({"email": email}, sort=[("_id", -1)]))
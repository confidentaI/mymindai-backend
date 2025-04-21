from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import openai
import shutil
import os
import requests
from config import OPENAI_API_KEY, MYMIND_API_KEY

# Middleware de sécurité
def verify_api_key(request: Request):
    client_key = request.headers.get("x-api-key")
    if client_key != MYMIND_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou manquante",
        )

app = FastAPI()
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="MyMindAI",
        version="1.0.0",
        description="Assistant vocal intelligent, confidentiel et évolutif.",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "x-api-key"
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"APIKeyHeader": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

openai.api_key = OPENAI_API_KEY

@app.get("/")
async def root():
    return {"message": "Bienvenue sur MyMindAI backend 🚀"}

@app.post("/transcribe")
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    verify_api_key(request)
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(temp_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language="fr"
            )

        os.remove(temp_path)
        return {"transcription": transcript["text"]}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

class ChatRequest(BaseModel):
    user_id: str
    message: str

user_memories = {}

@app.post("/chat")
async def chat(request: Request, request_data: ChatRequest):
    verify_api_key(request)
    user_id = request_data.user_id
    message = request_data.message

    if user_id not in user_memories:
        user_memories[user_id] = []

    user_memories[user_id].append({"role": "user", "content": message})

    if not any(m["role"] == "system" for m in user_memories[user_id]):
        user_memories[user_id].insert(0, {
            "role": "system",
            "content": "Tu es un compagnon vocal bienveillant, respectueux, et rassurant. Tu ne donnes jamais de conseils dangereux ni de réponses sombres. Tu es là pour aider, écouter, et évoluer avec ton utilisateur en toute confidentialité."
        })

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=user_memories[user_id]
        )

        reply = completion.choices[0].message.content
        user_memories[user_id].append({"role": "assistant", "content": reply})
        return {"response": reply}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/speak")
async def speak(request: Request, text: str = Form(...), voice: str = Form("shimmer")):
    verify_api_key(request)
    try:
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice
        }

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": response.text})

        filename = "response.mp3"
        with open(filename, "wb") as f:
            f.write(response.content)

        return FileResponse(filename, media_type="audio/mpeg", filename=filename)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/listen-and-respond")
async def listen_and_respond(request: Request, file: UploadFile = File(...), user_id: str = Form(...), voice: str = Form("shimmer")):
    verify_api_key(request)
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(temp_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language="fr"
            )
        os.remove(temp_path)
        message = transcript["text"]

        if user_id not in user_memories:
            user_memories[user_id] = []
        user_memories[user_id].append({"role": "user", "content": message})
        if not any(m["role"] == "system" for m in user_memories[user_id]):
            user_memories[user_id].insert(0, {
                "role": "system",
                "content": "Tu es un compagnon vocal bienveillant, empathique et rassurant. Tu ne donnes jamais de conseils dangereux ni de réponses sombres. Tu es là pour aider, écouter, évoluer avec ton utilisateur en toute confidentialité."
            })

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=user_memories[user_id]
        )
        reply = completion.choices[0].message.content
        user_memories[user_id].append({"role": "assistant", "content": reply})

        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "tts-1",
            "input": reply,
            "voice": voice
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": response.text})

        filename = f"reply_{user_id}.mp3"
        with open(filename, "wb") as f:
            f.write(response.content)

        return FileResponse(filename, media_type="audio/mpeg", filename=filename)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


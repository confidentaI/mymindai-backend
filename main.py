from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import openai
import shutil
import os
import requests
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MYMIND_API_KEY = os.getenv("MYMIND_API_KEY")
app = FastAPI()

openai.api_key = OPENAI_API_KEY

# Petite mémoire en RAM
user_memories = {}

# Middleware de sécurité
def verify_api_key(request: Request):
    client_key = request.headers.get("x-api-key")
    if client_key != MYMIND_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou manquante",
        )

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

@app.post("/chat")
async def chat(request: Request, body: ChatRequest):
    verify_api_key(request)
    user_id = body.user_id
    message = body.message

    if user_id not in user_memories:
        user_memories[user_id] = []

    user_memories[user_id].append({"role": "user", "content": message})

    if not any(m["role"] == "system" for m in user_memories[user_id]):
        user_memories[user_id].insert(0, {
            "role": "system",
            "content": "Tu es un compagnon vocal bienveillant et positif. Tu ne donnes jamais de conseils dangereux."
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
async def listen_and_respond(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    voice: str = Form("shimmer")
):
    verify_api_key(request)
    print("⚡ Début traitement /listen-and-respond")
    try:
        # 1. Enregistrement temporaire
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ Fichier temporaire enregistré : {temp_path}")

        # 2. Transcription Whisper
        with open(temp_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language="fr"
            )
        os.remove(temp_path)
        message = transcript["text"]
        print(f"✅ Transcription obtenue : {message}")

        # 3. Mise à jour mémoire utilisateur
        if user_id not in user_memories:
            user_memories[user_id] = []
        user_memories[user_id].append({"role": "user", "content": message})

        if not any(m["role"] == "system" for m in user_memories[user_id]):
            user_memories[user_id].insert(0, {
                "role": "system",
                "content": "Tu es un compagnon vocal bienveillant et positif. Tu ne donnes jamais de conseils dangereux."
            })

        # 4. Réponse GPT
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=user_memories[user_id]
        )
        reply = completion.choices[0].message.content
        user_memories[user_id].append({"role": "assistant", "content": reply})
        print(f"✅ Réponse GPT : {reply}")

        # 5. Synthèse TTS
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
            print(f"❌ Erreur synthèse vocale : {response.text}")
            return JSONResponse(status_code=500, content={"error": response.text})

        filename = f"reply_{user_id}.mp3"
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Fichier vocal généré : {filename}")

        return FileResponse(filename, media_type="audio/mpeg", filename=filename)

    except Exception as e:
        print(f"❌ Erreur globale dans /listen-and-respond : {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

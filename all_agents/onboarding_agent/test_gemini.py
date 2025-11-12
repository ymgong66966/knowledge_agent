import json
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request

PROJECT_ID = "withcare-onboarding"
LOCATION = "us-central1"
MODEL = "gemini-2.0-flash-lite-001"
ENDPOINT = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL}:generateContent"

KEY_PATH = "/Users/xyxg025/knowledge_agent/all_agents/onboarding_agent/withcare-onboarding-502adb724fa1.json"  # <-- your attached key path

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
creds = service_account.Credentials.from_service_account_file(KEY_PATH, scopes=SCOPES)
creds.refresh(Request())
token = creds.token

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
    # Helps ensure the right project is billed/authorized:
    "x-goog-user-project": PROJECT_ID,
}

payload = {
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "Write a friendly two-sentence greeting for caregivers."}]
        }
    ],
    "generationConfig": {
        "temperature": 0.7,
        "maxOutputTokens": 256,
        "topP": 0.95,
        "topK": 40
    }
    # Optional extras:
    # "systemInstruction": {"role":"system","parts":[{"text":"You are a concise care navigator."}]},
    # "safetySettings": [{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_LOW_AND_ABOVE"}]
}

resp = requests.post(ENDPOINT, headers=headers, data=json.dumps(payload), timeout=60)
resp.raise_for_status()
print(resp.json())
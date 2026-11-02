from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Test FastAPI Project")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    # هذا المسار سيوضح لنا فوراً إذا كان السيرفر قادراً على قراءة المفتاح البرمجي أم لا
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if api_key:
        return {
            "status": "success",
            "message": "FastAPI is working perfectly on Vercel!",
            "api_key_found": True,
            "key_preview": f"{api_key[:4]}...{api_key[-3:]}"  # يعرض أول واخر حروف للتأكد من سلامته
        }
    else:
        return {
            "status": "warning",
            "message": "FastAPI is working, but GOOGLE_API_KEY was NOT found in environment variables.",
            "api_key_found": False
        }

from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Company Gemini Cloud API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# نموذج طلب واحد يحتوي على السؤال والنص المستخرج (أو سياق الشركة) معاً
# هذا هو الأسلوب القياسي والآمن للـ Serverless لضمان عدم ضياع البيانات
class RAGRequest(BaseModel):
    question: str
    context: str        # يتم استخراج النص في واجهة الموقع (Frontend) وإرساله هنا لضمان استقرار السيرفر
    session_id: str

@app.get("/")
def home():
    return {"status": "FastAPI Serverless POST Handler is active!"}

@app.post("/chat/")
async def chat_with_company_bot(request: RAGRequest):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY missing from Vercel environment.")
        
    try:
        system_instruction = (
            "أنت موظف خدمة عملاء ذكي ومحترف لشركتنا. أجب على سؤال العميل بناءً على السياق المرفق فقط.\n"
            "1. كن مهذباً وواضحاً وسريع الإجابة.\n"
            "2. إذا لم تجد الإجابة في السياق المرفق، قل بلطف: 'عذراً يا فندم، لا أملك هذه المعلومة حالياً'.\n\n"
            f"السياق المستخرج من ملفات الشركة:\n{request.context}"
        )
        
        contents = [{
            "role": "user",
            "parts": [{"text": request.question}]
        }]
        
        url = f"https://googleapis.com{api_key}"
        
        payload = {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
            "generationConfig": {
                "temperature": 0.2
            }
        }
        
        # استخدام httpx مع إغلاق الاتصال فوراً لحماية الذاكرة ومنع الـ 500
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)
            response_json = response.json()
            
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response_json.get('error', {}).get('message', 'Google API Error')
            )
            
        response_text = response_json['candidates'][0]['content']['parts'][0]['text']
        return {"response": response_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

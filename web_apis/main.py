from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
import pypdf
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List

app = FastAPI(title="Company Cloud RAG API (REST Architecture)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    session_id: str 

@app.get("/")
def home():
    return {"status": "FastAPI Permanent REST Cloud RAG is active and running perfectly!"}

@app.post("/upload-document/")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="يجب أن يكون الملف بصيغة PDF فقط.")
    
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Supabase credentials missing in Vercel settings.")
        
    try:
        # 1. قراءة النصوص من ملف الـ PDF مباشرة من الذاكرة
        pdf_reader = pypdf.PdfReader(file.file)
        text_chunks = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                text_chunks.append(text)
                
        extracted_context = "\n".join(text_chunks)
        if not extracted_context.strip():
            raise ValueError("الملف المرفوع فارغ أو لا يحتوي على نصوص مقروءة.")
            
        # 2. حفظ النص في Supabase عبر الـ REST API مباشرة (Upsert)
        url = f"{supabase_url}/rest/v1/company_documents"
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"
        }
        payload = {
            "session_id": session_id,
            "context_text": extracted_context
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
            
        # تم إصلاح الخطأ المطبعي هنا بكتابة أكواد نجاح الاستجابة القياسية [200, 201] ليعمل المفسر بنجاح
        if response.status_code not in:
            raise HTTPException(status_code=response.status_code, detail=f"Database storage failure: {response.text}")
            
        return {"message": f"تمت معالجة ملف الشركة '{file.filename}' وحفظه سحابياً بنجاح!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء معالجة وحفظ المستند: {str(e)}")

@app.post("/chat/")
async def chat_with_customer(request: ChatRequest):
    google_key = os.environ.get("GOOGLE_API_KEY")
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_ANON_KEY")
    
    if not google_key or not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Required environment variables are missing from Vercel.")
        
    try:
        # 1. جلب سياق المستند من Supabase عبر الـ REST API
        doc_url = f"{supabase_url}/rest/v1/company_documents?session_id=eq.{request.session_id}&select=context_text"
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        
        with httpx.Client(timeout=15.0) as client:
            doc_res = client.get(doc_url, headers=headers)
            doc_data = doc_res.json()
            
        if not doc_data or len(doc_data) == 0:
            raise HTTPException(status_code=400, detail="برجاء رفع ملف الـ PDF الخاص بالشركة أولاً لتدريب الروبوت.")
            
        context = doc_data[0]["context_text"]
        
        # 2. جلب آخر 6 رسائل من تاريخ المحادثة من سوبابيز عبر الـ REST API
        hist_url = f"{supabase_url}/rest/v1/chat_history_store?session_id=eq.{request.session_id}&select=role,message_text&order=id.desc&limit=6"
        
        with httpx.Client(timeout=15.0) as client:
            hist_res = client.get(hist_url, headers=headers)
            history_data = reversed(hist_res.json())
            
        # 3. بناء توجيهات النظام والرسائل لـ Gemini
        system_instruction = (
            "أنت موظف خدمة عملاء ذكي ومحترف لشركتنا. أجب على سؤال العميل بناءً على السياق المرفق وتاريخ المحادثة فقط.\n"
            "1. كن مهذباً وواضحاً وسريع الإجابة.\n"
            "2. إذا لم تجد الإجابة في السياق المرفق، قل بلطف: 'عذراً يا فندم، لا أملك هذه المعلومة حالياً'.\n\n"
            f"السياق المستخرج من ملفات الشركة:\n{context}"
        )
        
        contents = []
        for msg in history_data:
            contents.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["message_text"]}]
            })
            
        contents.append({
            "role": "user",
            "parts": [{"text": request.question}]
        })
        
        # 4. إرسال الطلب إلى واجهة جوجل جيميناي الرسمية المستقرة لـ REST
        gemini_url = f"https://googleapis.com{google_key}"
        payload = {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
            "generationConfig": {
                "temperature": 0.2
            }
        }
        
        with httpx.Client(timeout=30.0) as client:
            gemini_res = client.post(gemini_url, json=payload)
            gemini_data = gemini_res.json()
            
        if gemini_res.status_code != 200:
            raise HTTPException(status_code=gemini_res.status_code, detail="Google AI API Service Failure.")
            
        response_text = gemini_data['candidates'][0]['content']['parts'][0]['text']
        
        # 5. حفظ الحوار الجديد في جدول التاريخ سحابياً لتحديث الذاكرة
        save_url = f"{supabase_url}/rest/v1/chat_history_store"
        chat_payload = [
            {"session_id": request.session_id, "role": "user", "message_text": request.question},
            {"session_id": request.session_id, "role": "model", "message_text": response_text}
        ]
        
        with httpx.Client(timeout=15.0) as client:
            client.post(save_url, json=chat_payload, headers=headers)
            
        return {"response": response_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

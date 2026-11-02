from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
import pypdf
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List

app = FastAPI(title="Company Gemini REST RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extracted_context = ""
is_document_uploaded = False
chat_histories: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    question: str
    session_id: str 

# @app.get("/")
# def home():
#     return {"status": "FastAPI Ultra-Light REST RAG is active and running perfectly!"}

@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    global extracted_context, is_document_uploaded
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="يجب أن يكون الملف بصيغة PDF فقط.")
    
    try:
        pdf_reader = pypdf.PdfReader(file.file)
        text_chunks = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                text_chunks.append(text)
                
        extracted_context = "\n".join(text_chunks)
        
        if not extracted_context.strip():
            raise ValueError("الملف المرفوع فارغ أو لا يحتوي على نصوص مقروءة.")
            
        is_document_uploaded = True
        return {"message": f"تمت قراءة ملف الشركة '{file.filename}' وحفظ سياقه بنجاح!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء معالجة المستند: {str(e)}")

# @app.post("/chat/")
# async def chat_with_customer(request: ChatRequest):
    global extracted_context, is_document_uploaded
    if not is_document_uploaded:
        raise HTTPException(status_code=400, detail="برجاء رفع ملف الـ PDF الخاص بالشركة أولاً لتدريب الروبوت.")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="لم يتم العثور على مفتاح الـ GOOGLE_API_KEY في السيرفر.")
        
    try:
        if request.session_id not in chat_histories:
            chat_histories[request.session_id] = []
            
        history = chat_histories[request.session_id]
        
        system_instruction = (
            "أنت موظف خدمة عملاء ذكي ومحترف لشركتنا. أجب على سؤال العميل بناءً على السياق المرفق وتاريخ المحادثة فقط.\n"
            "التزم بالقواعد التالية:\n"
            "1. كن مهذباً وواضحاً وسريع الإجابة.\n"
            "2. إذا لم تجد الإجابة في السياق المرفق، قل بلطف: 'عذراً يا فندم، لا أملك هذه المعلومة حالياً، هل يمكنك ترك رقمك ليتواصل معك الدعم الفني؟'.\n"
            "3. لا تقم باختراع أو تخمين أي معلومات غير موجودة في المستندات.\n\n"
            f"السياق المستخرج من ملفات الشركة:\n{extracted_context}"
        )
        
        # بناء المحتوى لـ Gemini REST API متوافق مع نموذج gemini-2.5-flash
        contents = []
        for msg in history[-6:]:
            contents.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["text"]}]
            })
            
        contents.append({
            "role": "user",
            "parts": [{"text": request.question}]
        })
        
        # الرابط المباشر لواجهة جوجل الرسمية المستقرة
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
        
        # إرسال الطلب عبر اتصال HTTP قياسي خفيف جداً
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)
            response_json = response.json()
            
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Google API Error: {response_json.get('error', {}).get('message', 'Unknown Error')}"
            )
            
        response_text = response_json['candidates'][0]['content']['parts'][0]['text']
        
        history.append({"role": "user", "text": request.question})
        history.append({"role": "model", "text": response_text})
        
        return {"response": response_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء معالجة الرد الذكي: {str(e)}")

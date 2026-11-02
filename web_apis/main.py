from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
import pypdf
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from supabase import create_client, Client

app = FastAPI(title="Company Cloud RAG API with Supabase")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# دالة الاتصال المحدثة لتتوافق مع الـ Integration التلقائي لـ Vercel
def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL")
    # تم تعديل الاسم هنا ليطابق الاسم المرفوع تلقائياً من نظام الربط
    key = os.environ.get("SUPABASE_ANON_KEY") 
    if not url or not key:
        raise HTTPException(status_code=500, detail="Supabase configuration keys are missing in Vercel.")
    return create_client(url, key)


class ChatRequest(BaseModel):
    question: str
    session_id: str 

@app.get("/")
def home():
    return {"status": "FastAPI Permanent Cloud RAG is active and running!"}

@app.post("/upload-document/")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="يجب أن يكون الملف بصيغة PDF فقط.")
    
    try:
        # 1. قراءة النصوص من ملف الـ PDF مباشرة
        pdf_reader = pypdf.PdfReader(file.file)
        text_chunks = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                text_chunks.append(text)
                
        extracted_context = "\n".join(text_chunks)
        if not extracted_context.strip():
            raise ValueError("الملف المرفوع فارغ أو لا يحتوي على نصوص مقروءة.")
            
        # 2. حفظ النص في قاعدة البيانات السحابية بربطه بالـ session_id
        supabase = get_supabase()
        supabase.table("company_documents").upsert({
            "session_id": session_id,
            "context_text": extracted_context
        }).execute()
        
        return {"message": f"تمت معالجة ملف الشركة '{file.filename}' وحفظه برقم الجلسة سحابياً بنجاح!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء معالجة وحفظ المستند: {str(e)}")

@app.post("/chat/")
async def chat_with_customer(request: ChatRequest):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY missing from Vercel environment.")
        
    try:
        supabase = get_supabase()
        
        # 1. جلب سياق ملف الـ PDF الدائم من Supabase
        doc_query = supabase.table("company_documents").select("context_text").eq("session_id", request.session_id).execute()
        if not doc_query.data:
            raise HTTPException(status_code=400, detail="برجاء رفع ملف الـ PDF الخاص بالشركة لهذه الجلسة أولاً.")
            
        context = doc_query.data[0]["context_text"]
        
        # 2. جلب آخر 6 رسائل من تاريخ المحادثة الفعلي المخزن سحابياً
        history_query = supabase.table("chat_history_store").select("role", "message_text").eq("session_id", request.session_id).order("id", desc=True).limit(6).execute()
        
        # ترتيب التاريخ تصاعدياً ليفهمه الذكاء الاصطناعي
        history_data = reversed(history_query.data)
        
        # 3. بناء نص النظام والتعليمات
        system_instruction = (
            "أنت موظف خدمة عملاء ذكي ومحترف لشركتنا. أجب على سؤال العميل بناءً على السياق المرفق وتاريخ المحادثة فقط.\n"
            "1. كن مهذباً وواضحاً وسريع الإجابة.\n"
            "2. إذا لم تجد الإجابة في السياق المرفق، قل بلطف: 'عذراً يا فندم، لا أملك هذه المعلومة حالياً'.\n\n"
            f"السياق المستخرج من ملفات الشركة:\n{context}"
        )
        
        # تجهيز الرسائل المرسلة لـ Gemini REST API
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
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)
            response_json = response.json()
            
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Google API Service Failure.")
            
        response_text = response_json['candidates']['content']['parts']['text']
        
        # 4. حفظ الحوار الجديد في قاعدة البيانات السحابية لتحديث الذاكرة بشكل مستدام
        supabase.table("chat_history_store").insert([
            {"session_id": request.session_id, "role": "user", "message_text": request.question},
            {"session_id": request.session_id, "role": "model", "message_text": response_text}
        ]).execute()
        
        return {"response": response_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# TUpXL4pTKh2d0XCd

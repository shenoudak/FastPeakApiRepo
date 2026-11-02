from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List
import pypdf
from google import genai
from google.genai import types

app = FastAPI(title="Company Gemini RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# متغيرات لتخزين سياق المستند المستخرج في الذاكرة
extracted_context = ""
is_document_uploaded = False

# قاموس لحفظ تاريخ المحادثات لكل جلسة (Session)
chat_histories: Dict[str, List[Dict[str, str]]] = {}

# دالة مساعدة لتهيئة عميل جوجل عند الحاجة فقط والتحقق من المفتاح
def get_gemini_client():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="خطأ في السيرفر: لم يتم العثور على GOOGLE_API_KEY في إعدادات البيئة."
        )
    return genai.Client(api_key=api_key)

class ChatRequest(BaseModel):
    question: str
    session_id: str 

@app.get("/")
def home():
    return {"status": "FastAPI Light Gemini RAG is active and running perfectly!"}

@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    global extracted_context, is_document_uploaded
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="يجب أن يكون الملف بصيغة PDF فقط.")
    
    try:
        # قراءة نصوص الـ PDF مباشرة من الذاكرة باستخدام pypdf الخفيفة
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

@app.post("/chat/")
async def chat_with_customer(request: ChatRequest):
    global extracted_context, is_document_uploaded
    if not is_document_uploaded:
        raise HTTPException(status_code=400, detail="برجاء رفع ملف الـ PDF الخاص بالشركة أولاً لتدريب الروبوت.")
    
    try:
        client = get_gemini_client()
        
        # إدارة تاريخ المحادثة للجلسة الحالية
        if request.session_id not in chat_histories:
            chat_histories[request.session_id] = []
            
        history = chat_histories[request.session_id]
        
        # بناء نص التوجيه والنظام المدمج مع السياق
        system_instruction = (
            "أنت موظف خدمة عملاء ذكي ومحترف لشركتنا. أجب على سؤال العميل بناءً على السياق المرفق وتاريخ المحادثة فقط.\n"
            "التزم بالقواعد التالية:\n"
            "1. كن مهذباً وواضحاً وسريع الإجابة.\n"
            "2. إذا لم تجد الإجابة في السياق المرفق، قل بلطف: 'عذراً يا فندم، لا أملك هذه المعلومة حالياً، هل يمكنك ترك رقمك ليتواصل معك الدعم الفني؟'.\n"
            "3. لا تقم باختراع أو تخمين أي معلومات غير موجودة في المستندات.\n\n"
            f"السياق المستخرج من ملفات الشركة:\n{extracted_context}"
        )
        
        # تجهيز الرسائل مع التاريخ لإرسالها لـ Gemini
        contents = []
        # إضافة آخر 6 رسائل من التاريخ للحفاظ على حجم الذاكرة في الـ Serverless
        for msg in history[-6:]:
            contents.append(types.Content(
                role=msg["role"],
                parts=[types.Part.from_text(text=msg["text"])]
            ))
            
        # إضافة السؤال الحالي للعميل
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=request.question)]
        ))
        
        # استدعاء نموذج جيميناي الخفيف والمحدث لعام 2026
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2
            )
        )
        
        # حفظ الحوار الحالي في التاريخ
        history.append({"role": "user", "text": request.question})
        history.append({"role": "model", "text": response.text})
        
        return {"response": response.text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء معالجة الرد الذكي: {str(e)}")

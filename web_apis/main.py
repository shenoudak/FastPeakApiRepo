from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Company RAG Chatbot API (Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# متغيرات عامة مبدئية بدون تهيئة ثقيلة في جذر الملف
vectorstore = None
is_document_uploaded = False
chat_histories: Dict[str, List] = {}

system_prompt = (
    "أنت موظف خدمة عملاء ذكي ومحترف لشركتنا. أجب على سؤال العميل بناءً على السياق المرفق وتاريخ المحادثة فقط.\n"
    "التزم بالقواعد التالية:\n"
    "1. كن مهذباً وواضحاً وسريع الإجابة.\n"
    "2. إذا لم تجد الإجابة في السياق المرفق، قل بلطف: 'عذراً يا فندم، لا أملك هذه المعلومة حالياً، هل يمكنك ترك رقمك ليتواصل معك الدعم الفني؟'.\n"
    "3. لا تقم باختراع أو تخمين أي معلومات غير موجودة في المستندات.\n\n"
    "السياق المستخرج من ملفات الشركة:\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

class ChatRequest(BaseModel):
    question: str
    session_id: str 

@app.get("/")
def home():
    return {"status": "FastAPI Cloud RAG is active and running!"}

@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore, is_document_uploaded
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="يجب أن يكون الملف بصيغة PDF فقط.")
    
    temp_file_path = f"/tmp/temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
        
    try:
        # استدعاء الحزم الثقيلة هنا فقط لتجنب استهلاك الذاكرة عند بدء السيرفر
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_core.vectorstores import InMemoryVectorStore
        
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
        splits = text_splitter.split_documents(docs)
        
        if not splits:
            raise ValueError("الملف المرفوع فارغ أو لا يحتوي على نصوص مقروءة.")
            
        # تهيئة المتجهات محلياً عند الرفع فقط
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = InMemoryVectorStore(embeddings)
        vectorstore.add_documents(splits)
        is_document_uploaded = True
        
        return {"message": f"تمت معالجة ملف الشركة '{file.filename}' بنجاح!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء معالجة المستند: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/chat/")
async def chat_with_customer(request: ChatRequest):
    global vectorstore, is_document_uploaded
    if not is_document_uploaded or vectorstore is None:
        raise HTTPException(status_code=400, detail="برجاء رفع ملف الـ PDF الخاص بالشركة أولاً لتدريب الروبوت.")
    
    try:
        # استدعاء نماذج المحادثة هنا بدلاً من بيئة التشغيل الأساسية لتقليل الحجم
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        
        if request.session_id not in chat_histories:
            chat_histories[request.session_id] = []
        
        history = chat_histories[request.session_id]
        
        # جلب النصوص المتطابقة
        retrieved_docs = vectorstore.similarity_search(request.question, k=3)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({
            "context": context, 
            "chat_history": history, 
            "question": request.question
        })
        
        history.append(HumanMessage(content=request.question))
        history.append(AIMessage(content=response))
        
        if len(history) > 10: 
            chat_histories[request.session_id] = history[-10:]
            
        return {"response": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء معالجة الرد الذكي: {str(e)}")

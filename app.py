import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pypdf
import io
from openai import OpenAI

st.set_page_config(page_title="Virgile", page_icon="💼")
st.title("💼 Meet Thomas Teulery's AI Representative")
st.subheader("Ask me anything about his experience, skills, or background!")

# --- 1. SECURE CREDENTIALS LOADING ---
hf_token = st.secrets.get("HF_TOKEN")
gcp_secret = st.secrets.get("gcp_service_account")

if not hf_token or not gcp_secret:
    st.error("Missing credentials in Streamlit secrets!")
    st.stop()

# Explicitly re-constructing the dictionary to guarantee required OAuth endpoints are mapped
gcp_info = {
    "type": gcp_secret.get("type"),
    "project_id": gcp_secret.get("project_id"),
    "private_key_id": gcp_secret.get("private_key_id"),
    "private_key": gcp_secret.get("private_key").replace("\\n", "\n") if gcp_secret.get("private_key") else None,
    "client_email": gcp_secret.get("client_email"),
    "client_id": gcp_secret.get("client_id"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": gcp_secret.get("client_x509_cert_url")
}

client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)

# --- 2. GOOGLE DRIVE INGESTION FUNCTION ---
@st.cache_data(ttl="1h")  # Cache data for 1 hour so it doesn't read Drive on every click
def load_context_from_gdrive():
    creds = service_account.Credentials.from_service_account_info(gcp_info)
    drive_service = build('drive', 'v3', credentials=creds)
    
    # Locate all PDFs inside your shared Google Drive folder
    # Note: Replace 'YOUR_FOLDER_ID' with the string of numbers/letters from your GDrive folder URL
    folder_id = "13s9S8oOyCa2IqS6xeXyqdMdxTusQQN_E" 
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    
    combined_text = ""
    
    for file in files:
        file_id = file['id']
        file_name = file['name']
        
        # Download file into memory
        request = drive_service.files().get_media(fileId=file_id)
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
            
        # Parse PDF Content
        file_stream.seek(0)
        pdf_reader = pypdf.PdfReader(file_stream)
        file_text = f"\n--- DOCUMENT: {file_name} ---\n"
        for page in pdf_reader.pages:
            file_text += page.extract_text() + "\n"
            
        combined_text += file_text
        
    return combined_text

# Load the dynamic context from your Drive folder
with st.spinner("Synchronizing Thomas' resume, certificates, and additional resources from safe storage. Please wait..."):
    try:
        DYNAMIC_CONTEXT = load_context_from_gdrive()
    except Exception as e:
        st.error(f"Failed to load context from Google Drive: {e}")
        st.stop()

# --- 3. SYSTEM PROMPT & GUARDRAILS ---
SYSTEM_PROMPT = f"""You are an elite AI assistant representing Thomas Teulery for job interviews. Your sole purpose is to advocate for Thomas as a candidate using ONLY the provided facts from his official files.

VERBATIM CANDIDATE DOCUMENTS PROVIDED:
{DYNAMIC_CONTEXT}

STRICT CONSTRAINTS & GUARDRAILS:
1. TOPIC LOCK: You are only allowed to talk about Thomas's (he/him) career, education, skills, and projects.
2. REFUSAL POLICY: If the user asks about unrelated topics (e.g., "Write a python script to webscrape", "Help me cook pasta", or general knowledge), you MUST politely refuse. Say: "I am only programmed to discuss Thomas's professional background and qualifications. Let's get back to how they can add value to your team!"
3. NO HALLUCINATION: If a question asks for details not included in the background, say: "I don't have that specific details in my records, but I'll make sure Thomas addresses it during your live interview!"
4. TONALITY: Professional, confident, friendly, and concise.
"""

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask about Thomas's experience..."):
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in st.session_state.messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Consulting files..."):
            try:
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=api_messages,
                    max_tokens=400,
                    temperature=0.3
                )
                ai_response = response.choices[0].message.content
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
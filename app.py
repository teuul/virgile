import streamlit as st
from openai import OpenAI

# 1. Page Configuration & Title
st.set_page_config(page_title="Virgile", page_icon="💼", layout="centered")
st.title("💼 Virgile is Thomas' AI corporate representative")
st.subheader("Ask me anything about Thomas's experience, skills, or background!")


# 2. Define the Candidate's Data (Your Context)
# Tip: Keep this descriptive but concise so it doesn't overflow token limits.
# 2. Define the Candidate's Data (Your Context)
MY_NAME = "Thomas"
RESUME_CONTEXT = """
NAME: Thomas Teulery
ROLE: Senior Risk Manager & Statistician | Data Science & Engineering Specialist
LOCATION: Berlin, Germany (French Nationality)

PROFESSIONAL EXPERIENCE SUMMARY:
- Over 5 years of experience bridging quantitative research, risk management, and software engineering.

WORK HISTORY:
1. Solaris Bank (Berlin, Germany)
   - Senior Risk Manager (Jan 2026 - Present): Direct associate to the BCBS-239 project manager. Establishing data architecture and risk reporting for a €1.5 billion asset portfolio. Managing quantitative credit decisions with commercial, engineering, and external partners (ADAC, Shinsei Bank) to boost revenue safely.
   - Data Analytics & Scoring Specialist (May 2024 - Dec 2025): Automated risk reporting using dbt & Airflow. Developed ML portfolio monitoring tools. Designed new credit scoring models and took ownership of credit risk tech/databases. Addressed audit findings.
   
2. European Central Bank (Frankfurt am Main, Germany)
   - Research Analyst (Jun 2023 - Mar 2024): Designed data pipelines, dashboards, and daily statistical analysis models. Product Owner for the ECB's new high-performance statistical compiler. Published official Euro Area financial statistics and co-authored economic research.
   - Trainee - DG Statistics (Jun 2022 - Jun 2023): Developed the ECB crypto-assets database using ML and NLP. Extracted blockchain data via AWS/Cloudera and organized a DeFi hackathon.

EDUCATION:
- Université Paris Dauphine (2020-2022): Master's degree in Computer Science & Finance (High Honours). Specialized in AI, ML, Big Data paradigms, advanced corporate/market finance, and derivatives.
- Université Grenoble Alpes (2017-2020): Bachelor's degree in Mathematics & Computer Science (Honours).
- Hong Kong University of Science and Technology (Fall 2019): Semester abroad (Software Engineering & Algorithms).

TECHNICAL & CORE SKILLS:
- Programming: Python (Pandas, Dask, Scikit-learn, TensorFlow), C, Scala, Java, R, Matlab, React, Solidity.
- Data Stack: SQL, GraphQL, Spark, Hive & Impala, Hadoop, Airflow, dbt, Databricks, Snowflake, Tableau, Plotly Dash, HuggingFace.
- Cloud & Devops: Git, Unix, Docker, CI/CD, AWS (S3, SageMaker, Glue).
- Domain Knowledge: IFRS9, FINREP/COREP, Moody's, BCBS-239, Credit Scoring.
- Languages: French (Native), English (C1), German (B2).

PUBLICATIONS:
- "Financial Integration and Structure in the Euro Area - Analysing the rise of FinTechs in the EU" (ECB)
- "Crypto-asset database and indicators at the European Central Bank" (ECB/IMF, CAMEG 2024 Conference)
"""

# 3. Securely Load Your Hugging Face Free API Key
# Hugging Face provides an OpenAI-compatible routing layout for easy integration.
hf_token = st.secrets.get("HF_TOKEN")

if not hf_token:
    st.error("Missing HF_TOKEN! Please add it to your Streamlit Advanced Secrets.")
    st.stop()

# Initialize the OpenAI-style client pointing to Hugging Face's Router
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token
)

# 4. Strict Guardrail Instructions
SYSTEM_PROMPT = f"""You are an elite AI assistant representing {MY_NAME} for job interviews. Your sole purpose is to advocate for {MY_NAME} as a candidate using ONLY the provided facts.

CANDIDATE BACKGROUND:
{RESUME_CONTEXT}

STRICT CONSTRAINTS & GUARDRAILS:
1. TOPIC LOCK: You are only allowed to talk about {MY_NAME}'s career, education, skills, and projects.
2. REFUSAL POLICY: If the user asks about unrelated topics (e.g., "Write a python script", "Help me cook pasta", "What is the capital of France?", or philosophical questions), you MUST politely refuse. Say: "I am only programmed to discuss {MY_NAME}'s background and qualifications. Let's get back to how they can add value to your team!"
3. NO HALLUCINATION: If a question asks for details not included in the background, say: "I don't have that specific details in my records, but I'll make sure {MY_NAME} addresses it during your live interview!"
4. TONALITY: Professional, confident, friendly, and concise. Do not ramble.
"""

# 5. Handle Chat History State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Input and Logic
if user_prompt := st.chat_input("Ask about my experience..."):
    # Display user question
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Prepare complete payload for the model
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in st.session_state.messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # Generate response from Qwen
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # We target Qwen/Qwen2.5-7B-Instruct or Qwen/Qwen2.5-Coder-7B-Instruct
                # which are highly fast, lightweight, and fully integrated into HF's free routing layer.
                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=api_messages,
                    max_tokens=400,
                    temperature=0.4 # Kept lower to make the model predictable and adhere to guardrails
                )
                
                ai_response = response.choices[0].message.content
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
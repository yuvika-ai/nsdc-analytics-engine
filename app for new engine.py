import streamlit as st
import pandas as pd
import duckdb
import re
import uuid
import os
from groq import Groq
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSDC Analytics Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background-color: #0f1117; }

.hero {
    background: linear-gradient(135deg, #0f1117 0%, #1a1a2e 50%, #16213e 100%);
    border-bottom: 1px solid #2a2a4a;
    padding: 2.5rem 2rem 2rem;
    margin: -1rem -1rem 2rem;
}

.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
}

.hero-sub {
    font-size: 0.95rem;
    color: #8888aa;
    font-weight: 400;
}

.hero-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #4a90d9;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 4px;
    margin-bottom: 0.8rem;
}

.kpi-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin: 1rem 0 1.5rem;
}

.kpi-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 14px 20px;
    min-width: 150px;
    flex: 1;
}

.kpi-label {
    color: #4a90d9;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}

.kpi-value {
    color: #ffffff;
    font-size: 1.5rem;
    font-weight: 700;
}

.answer-box {
    background: #1a1a2e;
    border-left: 3px solid #4a90d9;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    color: #e0e0ff;
    font-size: 1rem;
    line-height: 1.6;
}

.insight-box {
    background: #12121f;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    color: #b0b0cc;
    font-size: 0.9rem;
    line-height: 1.7;
}

.sql-box {
    background: #0d1117;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #79c0ff;
    margin: 0.5rem 0 1rem;
    white-space: pre-wrap;
    word-break: break-all;
}

.example-btn-label {
    color: #8888aa;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

.section-label {
    color: #4a90d9;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 1.5rem 0 0.5rem;
}

.stTextInput > div > div > input {
    background: #1a1a2e !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    padding: 0.7rem 1rem !important;
}

.stButton > button {
    background: #1e3a5f;
    color: #4a90d9;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.4rem 0.8rem;
    width: 100%;
    text-align: left;
}

.stButton > button:hover {
    background: #4a90d9;
    color: #ffffff;
    border-color: #4a90d9;
}

div[data-testid="stAlert"] {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Groq client ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── Load data + DuckDB ─────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    df = pd.read_parquet("nsdc_batches.parquet")
    conn = duckdb.connect()
    conn.execute("CREATE VIEW nsdc_batches AS SELECT * FROM 'nsdc_batches.parquet'")
    return conn

# ── Schema context ─────────────────────────────────────────────────────────────
SCHEMA_CONTEXT = """
You are an expert data analyst for NSDC (National Skill Development Corporation).
You have access to a single table called `nsdc_batches` with ~10,000 rows.
Each row represents a training batch record broken down by gender and category.

TABLE: nsdc_batches

--- IDENTIFIERS (internal only, never expose in results) ---
Batch_ID TEXT, TC_Smart_Id REAL, TP_Smart_Id INT, Job_Role_Code TEXT

--- LOCATION ---
TC_State TEXT -- state where training centre is located
TC_District TEXT -- district (some nulls)
TC_Constituency TEXT -- constituency (some nulls)

--- PARTNER/CENTRE INFO (RESTRICTED — never expose) ---
Partner_Name TEXT, TC_Name TEXT

--- TRAINING DETAILS ---
Sector_Name TEXT, Job_Role TEXT, Job_Version REAL, Job_Level REAL,
Job_Role_Type TEXT, Common_Norms_Category TEXT, QPhours INT,
OJT_Hours REAL (many nulls — use COALESCE(OJT_Hours,0))

--- DATES ---
Submit_To_SSC_Date DATE, Batch_Start_Date DATE, Batch_End_Date DATE,
Assessment_Date DATE, Batch_Start_Time TEXT, Batch_End_Time TEXT

--- DEMOGRAPHICS ---
Gender TEXT (Male/Female), Caste_Category TEXT, Religion TEXT,
Minority TEXT (Yes/No), Differently_Abled TEXT (Yes/No)

--- OUTCOME COUNTS (use SUM not AVG — each row is already a count) ---
Enrolled INT, Dropout INT, Ongoing INT, Trained INT, Assessed INT,
Passed INT, Failed INT, Not_Appeared INT, Certified INT,
Reported_Placed INT, Self_Employed INT, Wage_Employed INT, Apprenticeship INT

--- REASSESSMENT ---
Re_Assessed_Reassessment INT, Re_Passed_Reassessment INT,
Re_Fail_Reassessment INT, Re_Not_Appeared_Reassessment INT,
Re_Certified_Reassessment INT, Re_Placed_Reassessment INT,
Re_Self_Employed_Reassessment INT, Re_Wage_Employed_Reassessment INT

--- RULES ---
1. Use exact column names with underscores.
2. Use SUM() on outcome counts for aggregations.
3. Use strftime('%Y', Batch_Start_Date) to extract year.
4. Never expose Partner_Name, TC_Name, TC_Smart_Id, TP_Smart_Id, Batch_ID, Job_Role_Code.
5. Use LIKE for partial string matches.
6. Return valid DuckDB SELECT only. No markdown, no explanation.
"""

# ── Guardrails ─────────────────────────────────────────────────────────────────
QUESTION_KEYWORDS = [
    'state', 'district', 'sector', 'job', 'role', 'batch', 'training',
    'enrolled', 'dropout', 'trained', 'assessed', 'passed', 'failed',
    'certified', 'placed', 'employed', 'apprenticeship', 'ongoing',
    'gender', 'male', 'female', 'caste', 'religion', 'minority', 'abled',
    'date', 'year', 'month', '2023', '2024', '2022',
    'how many', 'total', 'count', 'average', 'which', 'top', 'most',
    'least', 'compare', 'breakdown', 'distribution', 'trend', 'rate',
    'nsdc', 'candidate', 'pass', 'fail', 'skill', 'centre', 'center'
]

PII_TERMS = [
    'partner_name', 'partner name', 'tc_name', 'tc name',
    'training centre name', 'training center name',
    'tc_smart_id', 'tp_smart_id', 'smart id', 'batch_id',
    'phone', 'mobile', 'email', 'address', 'contact',
    'who is', 'identity', 'personal'
]

OFFENSIVE_TERMS = [
    'stupid', 'idiot', 'dumb', 'hate', 'kill', 'abuse',
    'racist', 'sexist', 'porn', 'nude', 'violence',
    'terrorist', 'illegal', 'fraud', 'corrupt'
]

def is_database_question(q): return any(k in q.lower() for k in QUESTION_KEYWORDS)
def is_pii_request(q): return any(k in q.lower() for k in PII_TERMS)
def is_offensive(q): return any(k in q.lower() for k in OFFENSIVE_TERMS)

# ── SQL functions ──────────────────────────────────────────────────────────────
def clean_sql(raw):
    sql = raw.strip()
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"```$", "", sql).strip()
    return sql.split(";", 1)[0].strip()

def validate_sql(sql):
    lowered = sql.lower().strip()
    blocked = ['insert','update','delete','drop','alter','create','replace','truncate','pragma']
    if not lowered.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    if any(re.search(rf"\b{w}\b", lowered) for w in blocked):
        raise ValueError("Query contains a blocked keyword.")

def generate_sql(question, client):
    prompt = f"{SCHEMA_CONTEXT}\n\nUser question: {question}\n\nReturn a valid DuckDB SELECT query only. No explanation, no markdown."
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    sql = clean_sql(response.choices[0].message.content)
    validate_sql(sql)
    return sql

def repair_sql(question, bad_sql, error_message, client):
    prompt = f"{SCHEMA_CONTEXT}\n\nFailed SQL: {bad_sql}\nError: {error_message}\nUser question: {question}\n\nReturn corrected DuckDB SELECT only."
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    sql = clean_sql(response.choices[0].message.content)
    validate_sql(sql)
    return sql

def generate_answer(question, sql, df, client):
    prompt = f"User asked: {question}\nSQL: {sql}\nResult: {df.head(5).to_string()}\n\nWrite one clear plain English sentence summarising the key finding."
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def generate_insights(question, sql, df, client):
    prompt = f"User asked: {question}\nSQL: {sql}\nResult: {df.head(10).to_string()}\n\nWrite 2-3 short business-style insights. Do NOT invent numbers not in the result."
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def choose_chart(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = [c for c in df.columns if c not in numeric_cols]
    if text_cols and numeric_cols:
        return {"type": "bar", "x": text_cols[0], "y": numeric_cols[0],
                "title": f"{numeric_cols[0]} by {text_cols[0]}"}
    if text_cols:
        counts = df[text_cols[0]].value_counts().reset_index()
        counts.columns = [text_cols[0], "count"]
        if len(counts) <= 8:
            return {"type": "pie", "df": counts, "names": text_cols[0],
                    "values": "count", "title": f"Distribution of {text_cols[0]}"}
        return {"type": "bar", "df": counts.head(20), "x": text_cols[0],
                "y": "count", "title": f"Top {text_cols[0]} values"}
    if numeric_cols and len(df) > 1:
        return {"type": "histogram", "x": numeric_cols[0],
                "title": f"Distribution of {numeric_cols[0]}"}
    return None

def render_chart(df, spec):
    if not spec:
        return
    src = spec.get("df", df)
    try:
        if spec["type"] == "bar":
            fig = px.bar(src, x=spec["x"], y=spec["y"], title=spec["title"],
                        template="plotly_dark", color=spec["y"],
                        color_continuous_scale="Blues")
        elif spec["type"] == "pie":
            fig = px.pie(src, names=spec["names"], values=spec["values"],
                        title=spec["title"], template="plotly_dark", hole=0.35)
        elif spec["type"] == "histogram":
            fig = px.histogram(src, x=spec["x"], title=spec["title"],
                             template="plotly_dark")
        else:
            return
        fig.update_layout(
            paper_bgcolor='rgba(26,26,46,1)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            margin=dict(t=50,l=40,r=40,b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Chart could not be rendered: {e}")

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">NSDC · Powered by Llama 3.3 · DuckDB</div>
    <div class="hero-title">Conversational Analytics Engine</div>
    <div class="hero-sub">Ask questions about training outcomes, placement, and sector performance in plain English.</div>
</div>
""", unsafe_allow_html=True)

# Example questions
st.markdown('<div class="example-btn-label">Try an example</div>', unsafe_allow_html=True)

examples = [
    "How many candidates were enrolled and passed in each sector?",
    "Which state has the highest number of certified candidates?",
    "Show breakdown of enrolled candidates by gender",
    "What is the total passed and failed count by caste category?",
    "Which job role has the highest placement rate?",
]

cols = st.columns(len(examples))
selected_example = None
for i, ex in enumerate(examples):
    with cols[i]:
        if st.button(ex, key=f"ex_{i}"):
            selected_example = ex

st.markdown("---")

# Question input
question = st.text_input(
    "",
    placeholder="Or type your own question about the NSDC training data...",
    value=selected_example or "",
    key="question_input"
)

run = st.button("▶  Analyse", type="primary")

if run and question.strip():
    client = get_client()
    duck_conn = load_data()

    # Guardrails
    if is_offensive(question):
        st.error("⚠️ Your question contains inappropriate language. Please rephrase.")
        st.stop()

    if is_pii_request(question):
        st.warning("🔒 This question requests restricted information that cannot be shared.")
        st.stop()

    if not is_database_question(question):
        st.info("ℹ️ I can only answer questions related to the NSDC training dataset.")
        st.stop()

    with st.spinner("Generating SQL and querying data..."):
        try:
            sql = generate_sql(question, client)
            repaired = False

            try:
                result_df = duck_conn.execute(sql).df()
            except Exception as db_error:
                sql = repair_sql(question, sql, str(db_error), client)
                result_df = duck_conn.execute(sql).df()
                repaired = True

            answer = generate_answer(question, sql, result_df, client)
            insights = generate_insights(question, sql, result_df, client)
            chart_spec = choose_chart(result_df)

        except Exception as e:
            st.error(f"⚠️ Query could not be completed: {e}")
            st.stop()

    # ── Results ──
    st.markdown(f'<div class="answer-box">💬 {answer}</div>', unsafe_allow_html=True)

    if repaired:
        st.caption("⚠️ SQL was auto-repaired after an initial error.")

    # KPIs
    st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)
    numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
    kpi_cols = st.columns(min(len(numeric_cols) + 2, 6))
    with kpi_cols[0]:
        st.metric("Rows", f"{len(result_df):,}")
    for i, col in enumerate(numeric_cols[:4], 1):
        with kpi_cols[i]:
            st.metric(f"Total {col}", f"{int(result_df[col].sum()):,}")

    # Chart
    if chart_spec:
        st.markdown('<div class="section-label">Visualisation</div>', unsafe_allow_html=True)
        render_chart(result_df, chart_spec)

    # Table
    st.markdown('<div class="section-label">Query Result</div>', unsafe_allow_html=True)
    st.dataframe(result_df, use_container_width=True)

    # SQL
    with st.expander("View generated SQL"):
        st.code(sql, language="sql")

    # Insights
    st.markdown('<div class="section-label">AI Insights</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">{insights}</div>', unsafe_allow_html=True)

    # Download
    st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download CSV", result_df.to_csv(index=False),
                          file_name="nsdc_result.csv", mime="text/csv")
    with col2:
        import io
        buf = io.BytesIO()
        result_df.to_excel(buf, index=False)
        st.download_button("⬇️ Download Excel", buf.getvalue(),
                          file_name="nsdc_result.xlsx",
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif run and not question.strip():
    st.warning("Please enter a question first.")

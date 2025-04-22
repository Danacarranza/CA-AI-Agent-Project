import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from textblob import TextBlob
import joblib
from sklearn.ensemble import IsolationForest
import pulp
import yagmail
from dotenv import load_dotenv
from models import groq_chat

# Cargar variables desde .env
load_dotenv()

# ──────────────────────────────
# 1. Configuración de página y barra lateral
# ──────────────────────────────
st.set_page_config("AI‑HR Dashboard", layout="wide", page_icon="🎯")
st.title("🤖 AI‑Powered HR Dashboard for Decision‑Making")

st.sidebar.header("⚙️ Settings")
auto_send       = st.sidebar.checkbox("📧 Auto‑send report on upload", False)
email_to        = st.sidebar.text_input("Email recipient(s)", "hr@company.com")
training_budget = st.sidebar.number_input("📚 Training Hours Budget", 10, 500, 100)

if "emailed" not in st.session_state:
    st.session_state.emailed = False

# ──────────────────────────────
# 2. Carga de archivo Excel
# ──────────────────────────────
uploaded = st.file_uploader("📁 Upload HR Excel File (.xlsx)", type="xlsx")
if not uploaded:
    st.info("Upload your HR data file to start.")
    st.stop()

df = pd.read_excel(uploaded)
df["MonthNum"] = pd.to_datetime(df["Month"], format="%b", errors="coerce").dt.month
if "Comments" not in df.columns:
    df["Comments"] = ""

# Cargar modelo si existe
attr_model = None
if os.path.exists("attrition_model.joblib"):
    attr_model = joblib.load("attrition_model.joblib")

# ──────────────────────────────
# 3. Métricas clave
# ──────────────────────────────
st.subheader("📊 Key Metrics & Department Breakdown")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Attrition",    f"{df['Attrition Rate'].mean():.2%}")
c2.metric("Avg Training Hrs", f"{df['Training Hours'].mean():.1f}")
c3.metric("Avg Engagement",   f"{df['Engagement Score'].mean():.1f}")
c4.metric("Total Records",     len(df))

dept = df.groupby("Department")["Attrition Rate"].mean().reset_index()
fig_dept = px.bar(dept, x="Department", y="Attrition Rate",
                  labels={"Attrition Rate": "Rate"}, title="Attrition by Department")
st.plotly_chart(fig_dept, use_container_width=True)

# ──────────────────────────────
# 4. Detección de anomalías
# ──────────────────────────────
st.subheader("🚨 Anomaly Detection")
features = df[["Attrition Rate", "Training Hours", "Engagement Score"]].fillna(0)
iso = IsolationForest(contamination=0.05, random_state=42)
df["Anomaly"] = iso.fit_predict(features)
anom = df[df["Anomaly"] < 0]
if not anom.empty:
    st.warning(f"⚠️ {len(anom)} anomalous records detected:")
    st.dataframe(anom[["Employee ID", "Department", "Attrition Rate", "Training Hours"]])
else:
    st.success("✅ No major anomalies found.")

# ──────────────────────────────
# 5. Predicción de atrición con Prophet
# ──────────────────────────────
st.subheader("📈 Attrition Forecast (Next 3 Months)")
ts = (
    df.groupby("MonthNum")["Attrition Rate"]
      .mean().reset_index()
      .rename(columns={"MonthNum": "ds", "Attrition Rate": "y"})
)
ts["ds"] = pd.to_datetime(ts["ds"].apply(lambda m: f"2025-{int(m):02d}-01"))
m = Prophet(yearly_seasonality=True)
m.fit(ts)
future = m.make_future_dataframe(periods=3, freq="M")
fcst = m.predict(future)
st.plotly_chart(plot_plotly(m, fcst), use_container_width=True)
pred3 = fcst.tail(3)[["ds", "yhat"]].rename(columns={"ds": "Month", "yhat": "Predicted Attrition"})
st.table(pred3)

# ──────────────────────────────
# 6. Asignación óptima de entrenamiento
# ──────────────────────────────
st.subheader("⚙️ Prescriptive Training Allocation")
eng_by_dept = df.groupby("Department")["Engagement Score"].mean().to_dict()
prob = pulp.LpProblem("TrainAlloc", pulp.LpMaximize)
vars = {d: pulp.LpVariable(f"hrs_{d}", lowBound=0) for d in eng_by_dept}
prob += pulp.lpSum(eng_by_dept[d] * vars[d] for d in eng_by_dept)
prob += pulp.lpSum(vars[d] for d in eng_by_dept) <= training_budget
prob.solve()
alloc = {d: vars[d].value() for d in eng_by_dept}
st.bar_chart(pd.Series(alloc, name="Allocated Hrs"))

# ──────────────────────────────
# 7. Riesgo de rotación individual
# ──────────────────────────────
if attr_model:
    st.subheader("🤖 Employee Attrition Risk Scores")
    X = df.select_dtypes(include=np.number).fillna(0)
    df["Risk"] = attr_model.predict_proba(X)[:, 1]
    top_risk = df.nlargest(5, "Risk")[["Employee ID", "Department", "Risk"]]
    st.table(top_risk.style.format({"Risk": "{:.1%}"}))

# ──────────────────────────────
# 8. Análisis de comentarios con IA (Groq)
# ──────────────────────────────
st.subheader("💬 AI Feedback Analysis & Action Plan")
feedbacks = df["Comments"].astype(str).tolist()

if feedbacks and os.getenv("GROQ_API_KEY_"):
    themes = ["Workload", "Management", "Compensation", "Growth", "Well‑being", "Culture"]
    prompt_t = (
        "You are an HR analyst. Classify these comments by theme and count each:\n\n"
        + "\n".join(f"- {c}" for c in feedbacks[:20])
        + "\n\nThemes: " + ", ".join(themes)
        + "\n\nFormat as Theme: Count"
    )
    try:
        theme_res = groq_chat(prompt_t, max_tokens=150)
        st.markdown("**Theme counts:**\n" + theme_res.replace("\n", "  \n"))
    except Exception as e:
        st.error(f"❌ Theme error: {e}")

    prompt_s = (
        "You are an HR specialist. Given these comments, write a 2‑sentence summary "
        "of overall sentiment and key concerns:\n\n"
        + "\n".join(f"- {c}" for c in feedbacks[:20])
    )
    try:
        sentiment_res = groq_chat(prompt_s, max_tokens=150)
        st.markdown(f"**Sentiment summary:** {sentiment_res}")
    except Exception as e:
        st.error(f"❌ Sentiment error: {e}")

    prompt_a = (
        "Based on the themes and sentiment above, recommend the TOP 5 actionable HR initiatives "
        "to address these concerns, numbered 1–5."
    )
    try:
        actions_res = groq_chat(prompt_a, max_tokens=200)
        st.markdown("**Action plan:**\n" + actions_res.replace("\n", "  \n"))
    except Exception as e:
        st.error(f"❌ Actions error: {e}")
else:
    st.warning("⚠️ No comments or unset GROQ_API_KEY—skipping AI feedback.")

# ──────────────────────────────
# 9. Exportar y enviar por correo
# ──────────────────────────────
st.subheader("📥 Export & Email Report")
report = "HR_Report.xlsx"
df.to_excel(report, index=False)
st.download_button("⬇️ Download Excel Report", report)

def send_email():
    user = os.getenv("YAGMAIL_USER")
    pw = os.getenv("YAGMAIL_PASSWORD")
    if not user or not pw:
        st.error("🔒 Set YAGMAIL_USER & YAGMAIL_PASSWORD first")
        return
    try:
        yag = yagmail.SMTP(user, pw)
        yag.send(
            to=[e.strip() for e in email_to.split(",")],
            subject="📊 AI‑HR Dashboard Report",
            contents="Your latest HR dashboard report is attached.",
            attachments=[report]
        )
        st.success("📧 Report sent successfully!")
        st.session_state.emailed = True
    except Exception as e:
        st.error(f"❌ Email failed: {e}")

if st.button("✉️ Send Report Now"):
    send_email()
if auto_send and not st.session_state.emailed:
    send_email()

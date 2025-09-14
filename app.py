import streamlit as st
import pandas as pd
import PyPDF2
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# --- CONFIG PAGE ---
st.set_page_config(
    page_title="Agent IA de Préselection de CVs",
    page_icon="🤖",
    layout="wide"
)

# --- STYLE CSS ---
st.markdown("""
    <style>
        /* Global font and background */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f8fafc;
            color: #1a202c;
        }

        /* Titles */
        h1, h2, h3 {
            color: #2b6cb0;
            font-weight: bold;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #edf2f7;
        }

        /* Dataframe header */
        .dataframe th {
            background: #2b6cb0 !important;
            color: white !important;
            text-align: center !important;
        }

        /* Expander header */
        .streamlit-expanderHeader {
            font-weight: bold;
            color: #2b6cb0;
        }
    </style>
""", unsafe_allow_html=True)

# --- LOAD NLP MODEL ---
@st.cache_resource
def load_model():
    return spacy.load("processed_resumes")

nlp = spacy.load("en_core_web_lg")

# --- HEADER ---
st.markdown("<h1 style='text-align:center;'>✨ Agent IA de Préselection de CVs ✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:gray;'>Analyse automatique des CVs et correspondance avec une offre d’emploi</p>", unsafe_allow_html=True)
st.write("---")

# --- FUNCTIONS ---
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def extract_name_and_contact(text):
    name, email, phone = None, None, None
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    email_matches = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_matches:
        email = email_matches[0]

    phone_matches = re.findall(r'((?:\+216|00216)?\s?0?[2-9]\d{1}[-.\s]?\d{3}[-.\s]?\d{3,4})', text)
    if phone_matches:
        phone = phone_matches[0]

    return name, email, phone

# --- DATASET + MODEL ---
dataset = load_dataset("cnamuangtoun/resume-job-description-fit")
df = pd.DataFrame(dataset["train"])
df["text"] = df["resume_text"] + " " + df["job_description_text"]
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)
class_names = model.classes_

def predict_proba(text_instance):
    if isinstance(text_instance, str):
        text_instance = [text_instance]
    elif not isinstance(text_instance, list):
        text_instance = list(text_instance)
    vectorized_text = vectorizer.transform(text_instance)
    return model.predict_proba(vectorized_text)

def get_lime_explanation(text_to_explain):
    explainer = LimeTextExplainer(class_names=class_names)
    return explainer.explain_instance(
        text_to_explain,
        predict_proba,
        num_features=10
    )

# --- SIDEBAR ---
with st.sidebar:
    st.image('logo.png', use_container_width=True, output_format='PNG')
    st.subheader("📝 Description du poste")
    job_description = st.text_area(
        "Entrez la description du poste :",
        height=200,
        placeholder="Exemple : Développeur Python avec expérience en NLP..."
    )
    st.markdown("---")
    st.info("ℹ️ Cette application vous permet de :\n\n"
            "1️⃣ Télécharger plusieurs CVs (PDF)\n"
            "2️⃣ Extraire les informations clés\n"
            "3️⃣ Évaluer la correspondance avec un poste\n"
            "4️⃣ Identifier les meilleurs profils")

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📂 Télécharger les CVs")
    uploaded_files = st.file_uploader(
        "Choisissez les fichiers PDF",
        type="pdf",
        accept_multiple_files=True,
        help="Vous pouvez uploader plusieurs CVs à la fois."
    )

with col2:
    st.subheader("📊 Résultats de l'analyse")

    if uploaded_files and job_description:
        candidates = []
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
                name, email, phone = extract_name_and_contact(text)
                combined_text = text + " " + job_description
                proba = predict_proba(combined_text)
                fit_score = proba[0][1] if len(proba[0]) > 1 else proba[0][0]

                text_vec = vectorizer.transform([text])
                job_vec = vectorizer.transform([job_description])
                sim_score = cosine_similarity(text_vec, job_vec)[0][0]

                candidates.append({
                    "Nom": name or "Inconnu",
                    "Email": email or "Non trouvé",
                    "Téléphone": phone or "Non trouvé",
                    "Fit Score": fit_score,
                    "Similarity Score": sim_score,
                    "Combined Text": combined_text
                })

        if candidates:
            df = pd.DataFrame(candidates).sort_values(by="Fit Score", ascending=False)
            st.dataframe(df, use_container_width=True)

            st.markdown("### 🏅 Classement des candidats")
            for idx, candidate in df.iterrows():
                with st.expander(f"👤 {candidate['Nom']} | 🔹 Fit: {candidate['Fit Score']:.2f} | 🔹 Similarité: {candidate['Similarity Score']:.2f}"):
                    col_info, col_contact = st.columns([2, 1])

                    with col_info:
                        st.write(f"**📊 Fit Score :** {candidate['Fit Score']:.3f}")
                        if st.button(f"Voir l'explication pour {candidate['Nom']}", key=f"explain_{idx}"):
                            with st.spinner("Génération de l'explication..."):
                                try:
                                    explanation = get_lime_explanation(candidate['Combined Text'])
                                    st.subheader("🧩 LIME - Explication")
                                    st.write("Caractéristiques clés influençant la prédiction :")
                                    explanation_text = explanation.as_list()
                                    for feature, weight in explanation_text:
                                        color = "green" if weight > 0 else "red"
                                        st.markdown(f"<span style='color:{color}; font-weight:bold'>{feature}: {weight:.3f}</span>", unsafe_allow_html=True)
                                    explanation_html = explanation.as_html()
                                    st.components.v1.html(explanation_html, height=400, scrolling=True)
                                except Exception as e:
                                    st.error(f"Erreur lors de la génération de l'explication : {str(e)}")

                    with col_contact:
                        st.write("**📧 Email :**", candidate['Email'])
                        st.write("**📞 Téléphone :**", candidate['Téléphone'])

            st.success(f"📑 Total des candidats : {len(df)} | ✅ Score moyen : {df['Fit Score'].mean():.2f} | 🏆 Top : {df.iloc[0]['Nom']}")

    elif not uploaded_files:
        st.warning("📂 Veuillez télécharger des fichiers CV pour commencer.")
    elif not job_description:
        st.warning("📝 Veuillez entrer une description de poste dans la barre latérale.")

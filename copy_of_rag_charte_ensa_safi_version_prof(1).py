# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import faiss
import streamlit as st

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ENSA Safi — Assistant RAG",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #12355b;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #64748b;
        margin-bottom: 30px;
    }

    .bot-message {
        background: white;
        padding: 18px;
        border-radius: 14px;
        border-left: 5px solid #2563eb;
        margin: 10px 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.06);
    }

    .user-message {
        background: #eaf2ff;
        padding: 15px;
        border-radius: 14px;
        margin: 10px 0;
    }

    .source-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
        margin-top: 8px;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 40px;
        padding: 20px;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🎓 Assistant RAG — Charte ENSA Safi</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Posez vos questions et obtenez des réponses fondées uniquement sur la Charte ENSA Safi.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.markdown("### 📄 Document")

    uploaded_file = st.file_uploader(
        "Importer la Charte ENSA Safi",
        type=["pdf"]
    )

    st.markdown("---")

    st.markdown("### 🤖 Paramètres RAG")

    k = st.slider(
        "Nombre de passages récupérés",
        min_value=1,
        max_value=10,
        value=4
    )

    st.markdown("---")

    st.markdown("### 🔑 Groq API")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Votre clé API Groq"
    )

    st.markdown("---")

    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# PDF LOADING
# =========================================================

@st.cache_data
def extraire_pdf(pdf_bytes):

    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages = []

    for numero, page in enumerate(reader.pages, start=1):

        texte = page.extract_text() or ""

        texte = re.sub(
            r"\s+",
            " ",
            texte
        ).strip()

        pages.append({
            "page": numero,
            "texte": texte
        })

    return pages


# =========================================================
# CHUNKING
# =========================================================

def decouper_texte(
    texte,
    taille=900,
    chevauchement=150
):

    morceaux = []

    debut = 0

    while debut < len(texte):

        fin = min(
            debut + taille,
            len(texte)
        )

        if fin < len(texte):

            espace = texte.rfind(
                " ",
                debut,
                fin
            )

            if espace > debut:
                fin = espace

        morceau = texte[
            debut:fin
        ].strip()

        if morceau:
            morceaux.append(morceau)

        if fin == len(texte):
            break

        debut = max(
            fin - chevauchement,
            debut + 1
        )

    return morceaux


# =========================================================
# CREATE CHUNKS
# =========================================================

def creer_chunks(pages):

    chunks = []

    for page in pages:

        morceaux = decouper_texte(
            page["texte"]
        )

        for i, texte_chunk in enumerate(
            morceaux,
            start=1
        ):

            chunks.append({
                "page": page["page"],
                "chunk": i,
                "texte": texte_chunk
            })

    return chunks


# =========================================================
# BUILD FAISS INDEX
# =========================================================

@st.cache_resource
def construire_index(chunks):

    modele = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    textes_chunks = [
        c["texte"]
        for c in chunks
    ]

    vecteurs = modele.encode(
        textes_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    index = faiss.IndexFlatIP(
        vecteurs.shape[1]
    )

    index.add(vecteurs)

    return modele, index


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    if (
        st.session_state.get("processed_file")
        != uploaded_file.name
    ):

        with st.spinner(
            "📚 Analyse de la Charte ENSA Safi..."
        ):

            pages = extraire_pdf(pdf_bytes)

            chunks = creer_chunks(pages)

            modele, index = construire_index(
                chunks
            )

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.embedding_model = modele
            st.session_state.processed_file = uploaded_file.name

        st.success(
            f"✅ Document chargé : {len(pages)} pages "
            f"et {len(chunks)} chunks créés."
        )


# =========================================================
# SEARCH
# =========================================================

def rechercher(
    question,
    k=4
):

    modele = st.session_state.embedding_model

    index = st.session_state.index

    chunks = st.session_state.chunks

    vecteur_question = modele.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        vecteur_question,
        k
    )

    resultats = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx != -1:

            item = chunks[int(idx)].copy()

            item["score"] = float(score)

            resultats.append(item)

    return resultats


# =========================================================
# PROMPT
# =========================================================

def construire_prompt(
    question,
    passages
):

    contexte = "\n\n".join(

        f'[Source : page {p["page"]}]\n'
        f'{p["texte"]}'

        for p in passages
    )

    return f"""
Tu es un assistant de l'ENSA Safi.

Réponds uniquement à partir du CONTEXTE fourni.

Règles obligatoires :

1. Donne une réponse courte, claire et fidèle au document.
2. Cite toujours la ou les pages sous la forme (page X).
3. Si l'information exacte n'est pas présente dans le contexte,
   réponds exactement :

« Information non précisée dans la charte. »

4. N'invente jamais une information.
5. N'utilise aucune connaissance extérieure.
6. Ne déduis pas une information qui n'est pas explicitement présente.

CONTEXTE :

{contexte}

QUESTION :

{question}

RÉPONSE :
"""


# =========================================================
# GROQ RESPONSE
# =========================================================

def repondre(
    question,
    k=4
):

    passages = rechercher(
        question,
        k
    )

    prompt = construire_prompt(
        question,
        passages
    )

    client = Groq(
        api_key=api_key
    )

    completion = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )

    texte_reponse = (
        completion
        .choices[0]
        .message
        .content
    )

    pages_sources = sorted(
        {
            p["page"]
            for p in passages
        }
    )

    return (
        texte_reponse,
        passages,
        pages_sources
    )


# =========================================================
# CHECK SYSTEM STATUS
# =========================================================

if st.session_state.chunks is None:

    st.info(
        "👈 Commencez par importer la Charte ENSA Safi depuis la barre latérale."
    )

elif not api_key:

    st.warning(
        "🔑 Ajoutez votre Groq API Key dans la barre latérale pour commencer."
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="user-message">
                <strong>👤 Vous</strong><br>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="bot-message">
                <strong>🤖 Assistant ENSA</strong><br><br>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        if "sources" in message:

            with st.expander(
                "📚 Voir les sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source-box">
                            <strong>📄 Page {source["page"]}</strong>
                            — Score : {source["score"]:.3f}

                            <br><br>

                            {source["texte"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Posez votre question sur la Charte ENSA Safi..."
)


if question:

    if st.session_state.chunks is None:

        st.error(
            "❌ Veuillez d'abord importer le PDF."
        )

        st.stop()

    if not api_key:

        st.error(
            "❌ Veuillez ajouter votre Groq API Key."
        )

        st.stop()

    # USER MESSAGE

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })

    with st.chat_message("user"):
        st.write(question)

    # ASSISTANT

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Recherche dans la charte..."
        ):

            try:

                response, passages, pages_sources = repondre(
                    question,
                    k=k
                )

                st.markdown(
                    response
                )

                # SOURCES

                with st.expander(
                    f"📚 Sources — pages {pages_sources}"
                ):

                    for p in passages:

                        st.markdown(
                            f"""
                            **📄 Page {p["page"]}**
                            
                            Score : `{p["score"]:.3f}`

                            {p["texte"]}
                            """
                        )

                # SAVE MESSAGE

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": response,

                    "sources": passages

                })

            except Exception as e:

                st.error(
                    f"❌ Erreur : {e}"
                )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🎓 ENSA Safi — RAG Assistant
        <br>
        YaneCode Academy • AI Générative
    </div>
    """,
    unsafe_allow_html=True
)
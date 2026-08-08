r"""
Assistant RAG Maths CIV -- interface Streamlit.

Usage :
    uv add streamlit
    uv run streamlit run app.py

Prérequis dans le même dossier :
    - moteur_rag.py, filtre_niveau.py
    - resultats_pdf_2023.json (le corpus des 14 niveaux)
"""
import streamlit as st

from filtre_niveau import NIVEAU_LABELS
from moteur_rag import MoteurRAG, generer_reponse

CHEMIN_JSON = "resultats_pdf_2023.json"

# Interrupteur pour la mise en ligne publique (Streamlit Community Cloud).
# Ollama (Qwen 3B, Mistral 7B) n'est PAS disponible sur leurs serveurs, et
# même Qwen2.5-0.5B via transformers est risqué avec seulement ~1 Go de RAM
# alloué sur le plan gratuit -- les poids du modèle à eux seuls en
# consomment déjà la majeure partie. En mode consultation, l'appli reste
# 100% fiable (aucun appel LLM) : elle affiche directement le contenu exact
# du programme trouvé, sans reformulation -- ce qui sert justement l'objectif
# de "maîtrise du contenu" sans aucun risque d'hallucination.
# Remets à True si tu déploies un jour sur un hébergement avec plus de RAM.
AUTORISER_GENERATION_LLM = False

st.set_page_config(page_title="Assistant Maths CIV", page_icon="📐", layout="centered")


@st.cache_resource(show_spinner="Chargement du corpus des programmes...")
def charger_moteur():
    return MoteurRAG(CHEMIN_JSON)


def init_etat():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # historique affiché à l'écran
    if "historique_modele" not in st.session_state:
        st.session_state.historique_modele = []  # historique passé au modèle (léger)
    if "niveau" not in st.session_state:
        st.session_state.niveau = "3e"


init_etat()

try:
    moteur = charger_moteur()
except FileNotFoundError:
    st.error(
        f"Fichier introuvable : {CHEMIN_JSON}. "
        "Place-le dans le même dossier que app.py avant de relancer."
    )
    st.stop()

# ----------------------------------------------------------------------
# Barre latérale : réinitialisation + choix du modèle de génération
# (le choix du NIVEAU a été déplacé au-dessus de la zone de question,
#  voir plus bas, pour rester visible juste avant de poser la question)
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Paramètres")

    if st.button("🗑️ Nouvelle conversation"):
        st.session_state.messages = []
        st.session_state.historique_modele = []
        st.rerun()

    st.divider()
    if AUTORISER_GENERATION_LLM:
        st.subheader("Modèle de génération")
        moteur_generation = st.radio(
            "Moteur",
            options=["local", "ollama_qwen3b", "ollama_mistral"],
            format_func=lambda x: {
                "local": "Qwen2.5-0.5B (local, très rapide, faible)",
                "ollama_qwen3b": "Qwen2.5-3B (Ollama, bon compromis)",
                "ollama_mistral": "Mistral 7B (Ollama, plus lent)",
            }[x],
            index=1,  # Qwen 3B par défaut : meilleur compromis identifié jusqu'ici
            help=(
                "Ollama doit tourner en arrière-plan (icône barre système) pour les "
                "options Qwen 3B et Mistral 7B."
            ),
        )
        _NOMS_OLLAMA = {
            "ollama_qwen3b": "qwen2.5:3b-instruct-q4_K_M",
            "ollama_mistral": "mistral:7b-instruct-q4_K_M",
        }
    else:
        moteur_generation = None
        st.subheader("Mode")
        st.info(
            "📖 **Mode consultation** : les réponses affichent directement le "
            "contenu exact du programme officiel trouvé, sans reformulation "
            "par un modèle de génération."
        )

st.title("📐 Assistant Maths CIV")

# ----------------------------------------------------------------------
# Affichage de l'historique de conversation
# ----------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources du programme utilisées"):
                for s in msg["sources"]:
                    niv = NIVEAU_LABELS.get(s["niveau"], s["niveau"])
                    st.markdown(f"**[{niv}] {s['lecon_titre']}** — {s['texte']}")

# ----------------------------------------------------------------------
# Sélection du niveau, juste au-dessus de la zone de question
# ----------------------------------------------------------------------
niveau_choisi = st.selectbox(
    "Niveau de l'élève",
    options=list(NIVEAU_LABELS.keys()),
    format_func=lambda k: NIVEAU_LABELS[k],
    index=list(NIVEAU_LABELS.keys()).index(st.session_state.niveau),
    help=(
        "Les réponses ne s'appuient que sur le programme officiel du niveau "
        "choisi et des niveaux antérieurs de sa filière."
    ),
)
if niveau_choisi != st.session_state.niveau:
    # Le périmètre de contenu change avec le niveau : on repart sur une
    # conversation neuve pour éviter de mélanger du contexte d'un autre
    # niveau dans l'historique envoyé au modèle.
    st.session_state.niveau = niveau_choisi
    st.session_state.messages = []
    st.session_state.historique_modele = []
    st.rerun()

# ----------------------------------------------------------------------
# Nouvelle question
# ----------------------------------------------------------------------
question = st.chat_input("Pose ta question (ex. : comment résoudre f' + 2f = 0 ?)")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Questions de comptage/liste de leçons : réponse directe depuis les
        # données, sans jamais passer par le LLM (aucun modèle, même gros,
        # ne "compte" fiablement ce qu'il ne voit pas en entier).
        reponse_structurelle = moteur.repondre_structurelle(
            question, st.session_state.niveau
        )
        if reponse_structurelle is None:
            reponse_structurelle = moteur.repondre_position_lecon(
                question, st.session_state.niveau
            )
        if reponse_structurelle is None:
            reponse_structurelle = moteur.repondre_lecon_complete(
                question, st.session_state.niveau
            )
        if reponse_structurelle is None:
            reponse_structurelle = moteur.repondre_genealogie(
                question, st.session_state.niveau
            )
        if reponse_structurelle is not None:
            st.markdown(reponse_structurelle)
            st.caption("Réponse calculée directement depuis les données du programme, sans modèle de génération.")
            st.session_state.messages.append(
                {"role": "assistant", "content": reponse_structurelle, "sources": None}
            )
            st.session_state.historique_modele.append({"role": "user", "content": question})
            st.session_state.historique_modele.append(
                {"role": "assistant", "content": reponse_structurelle}
            )
            st.stop()

        with st.spinner("Recherche dans le programme..."):
            resultats = moteur.rechercher(question, st.session_state.niveau, k=5)
            contexte = moteur.contexte_pour_prompt(resultats)

        if not resultats:
            reponse = (
                "Je n'ai rien trouvé de pertinent dans le programme officiel pour "
                f"ce niveau ({NIVEAU_LABELS[st.session_state.niveau]}) sur cette question. "
                "Essaie de reformuler, ou vérifie que le niveau sélectionné est le bon."
            )
            st.markdown(reponse)
        elif not AUTORISER_GENERATION_LLM:
            # Mode consultation : on affiche directement le contenu trouvé,
            # organisé par leçon, sans jamais appeler de modèle.
            lignes = []
            derniere_lecon = None
            for r in resultats:
                if r["lecon_titre"] != derniere_lecon:
                    niv = NIVEAU_LABELS.get(r["niveau"], r["niveau"])
                    lignes.append(f"\n**[{niv}] {r['lecon_titre']}**")
                    derniere_lecon = r["lecon_titre"]
                lignes.append(f"- {r['texte']}")
            reponse = "\n".join(lignes)
            st.markdown(reponse)
        else:
            with st.spinner("Génération de la réponse (peut prendre un moment)..."):
                try:
                    if moteur_generation == "local":
                        reponse = generer_reponse(
                            question,
                            contexte,
                            historique=st.session_state.historique_modele,
                            moteur_generation="local",
                        )
                    else:
                        reponse = generer_reponse(
                            question,
                            contexte,
                            historique=st.session_state.historique_modele,
                            moteur_generation="ollama",
                            nom_modele=_NOMS_OLLAMA[moteur_generation],
                        )
                except Exception as e:
                    reponse = f"⚠️ Erreur inattendue pendant la génération : {e}"
            st.markdown(reponse)
            with st.expander("Sources du programme utilisées"):
                for r in resultats:
                    niv = NIVEAU_LABELS.get(r["niveau"], r["niveau"])
                    st.markdown(f"**[{niv}] {r['lecon_titre']}** — {r['texte']}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reponse,
            # En mode consultation, les sources sont déjà affichées en clair
            # dans le texte de la réponse -- pas besoin de les redupliquer
            # dans un expander séparé lors du réaffichage de l'historique.
            "sources": resultats if AUTORISER_GENERATION_LLM else None,
        }
    )
    # Historique "léger" transmis au modèle : uniquement question/réponse,
    # sans les sources (le modèle n'en a pas besoin, ça alourdirait le prompt
    # inutilement pour un petit modèle local).
    st.session_state.historique_modele.append({"role": "user", "content": question})
    st.session_state.historique_modele.append({"role": "assistant", "content": reponse})

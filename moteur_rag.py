r"""
Moteur de recherche RAG "filtré par niveau" pour l'assistant Maths CIV.

Pipeline :
  1. On calcule les niveaux autorisés pour le niveau cible de la question
     (filtre_niveau.niveaux_autorises) -- collège commun + parcours standard
     de la filière (voir filtre_niveau.py pour le détail).
  2. On restreint le corpus (resultats_pdf_2023.json) à ces niveaux-là.
  3. On indexe ce sous-corpus avec TF-IDF (zéro téléchargement, rapide,
     largement suffisant sur un corpus de cette taille -- qq centaines
     d'habiletés/contenus, très riche en vocabulaire technique distinctif).
  4. On retourne les K passages les plus pertinents pour la question.

Ce module ne fait AUCUN appel à un LLM : il s'arrête à la récupération du
contexte. Le prompt + l'appel à Ollama viennent ensuite (voir generer_reponse
en bas, qui est un ajout optionnel, à activer une fois Qwen/Mistral prêt).

Chaque "document" indexé correspond à UNE habileté (pas une leçon entière) :
c'est le niveau de granularité le plus utile pour répondre précisément à une
question ("comment dérive-t-on une fonction composée ?") sans noyer le modèle
dans tout le contenu d'une leçon de 58 pages.

Usage :
    uv run python .\moteur_rag.py
"""
import json
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from filtre_niveau import niveaux_autorises, NIVEAU_LABELS

# Rang "année scolaire" (indépendant de la filière), utilisé uniquement pour
# ordonner l'affichage de la généalogie d'une notion du plus élémentaire au
# plus avancé. Plusieurs niveaux peuvent partager le même rang (ex. 2ndeA et
# 2ndeC sont la même année, deux filières différentes).
ANNEE_RANG = {
    "6e": 0, "5e": 1, "4e": 2, "3e": 3,
    "2A": 4, "2C": 4,
    "1A1": 5, "1A2": 5, "1C": 5, "1D": 5,
    "TA1": 6, "TA2": 6, "TC": 6, "TD": 6,
}

# Bonus de score ajouté quand le verbe de la question (ex. "Résous...",
# "Démontre que...") correspond au verbe d'habileté d'un document (ex.
# "Résoudre", "Démontrer"). Dans le système ivoirien, les énoncés
# d'exercices sont formulés à partir de ces verbes -- c'est donc un vrai
# signal pédagogique, qu'on ajoute EN PLUS du score de contenu (TF-IDF),
# jamais mélangé dans le même texte vectorisé : sinon un document très
# court qui ne contient quasiment que le verbe prend un score anormalement
# élevé (artefact de normalisation du TF-IDF sur des textes courts), même
# s'il n'a aucun rapport avec le sujet de la question.
BONUS_VERBE = 0.2


def _stem_verbe(mot: str) -> str:
    """
    Racine grossière d'un verbe, pour rapprocher une forme conjuguée de la
    question ("Résous", "Calcule", "Démontre") de l'infinitif utilisé comme
    nom d'habileté dans le corpus ("Résoudre", "Calculer", "Démontrer").
    Heuristique simple (accents retirés, 4 premiers caractères) -- pas une
    vraie lemmatisation, mais suffisante pour ce vocabulaire limité et
    largement régulier.
    """
    t = unicodedata.normalize("NFKD", mot)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.lower()[:4]


# Liste des verbes d'habileté rencontrés dans les programmes DPFC (reprise
# de pdf_parser.py), utilisée pour repérer le(s) verbe(s) présent(s) dans
# la question posée par l'utilisateur.
VERBES_HABILETE = {
    "identifier", "connaitre", "connaître", "reconnaitre", "reconnaître",
    "noter", "nommer", "tracer", "construire", "justifier", "determiner",
    "déterminer", "calculer", "comparer", "resoudre", "résoudre", "traiter",
    "utiliser", "representer", "représenter", "interpreter", "interpréter",
    "demontrer", "démontrer", "etudier", "étudier", "decomposer",
    "décomposer", "placer", "reduire", "réduire", "ecrire", "écrire",
    "traduire", "etablir", "établir", "lineariser", "linéariser",
    "conjecturer", "savoir", "classer", "realiser", "réaliser",
    "denombrer", "dénombrer", "decrire", "décrire", "mesurer", "graduer",
    "lire", "exprimer", "trouver",
}
STEMS_HABILETE = {_stem_verbe(v) for v in VERBES_HABILETE}


def normaliser(texte: str) -> str:
    """Nettoyage léger avant vectorisation (accents conservés -- le TF-IDF
    français gère très bien les accents tels quels, pas besoin de les retirer)."""
    return re.sub(r"\s+", " ", texte).strip().lower()


def construire_documents(lecons: list) -> list:
    """
    Éclate chaque leçon en un document par habileté (granularité fine).
    Chaque document garde ses métadonnées (niveau, compétence, thème, leçon,
    habileté) pour pouvoir citer précisément la source dans la réponse.
    """
    documents = []
    for lecon in lecons:
        titre = lecon.get("lecon_titre", "")
        for h in lecon.get("habiletes", []):
            contenus = " ; ".join(h.get("contenus", []))
            # IMPORTANT : le verbe d'habileté (Résoudre, Calculer, Connaître,
            # Traiter, ...) n'est PAS inclus dans le texte vectorisé. Ces
            # verbes reviennent dans quasiment toutes les leçons du corpus et
            # n'apportent aucune information sur le SUJET -- pire, sur un
            # contenu très court (ex. "Résoudre : un exercice"), le TF-IDF
            # normalisé leur donne un poids disproportionné qui peut faire
            # remonter une leçon hors-sujet juste parce qu'elle partage ce
            # verbe générique avec la question. Seuls le titre de la leçon
            # (répété pour plus de poids) et le contenu réel (les notions,
            # formules) servent à la recherche ; le verbe reste affiché dans
            # "texte" pour l'utilisateur, mais hors du calcul de similarité.
            texte_recherche = f"{titre} {titre} : {contenus}"
            documents.append({
                "texte": f"{h['habilete']} : {contenus}",  # affichage complet
                "texte_norm": normaliser(texte_recherche),  # utilisé pour la recherche
                "niveau": lecon.get("niveau"),
                "competence": lecon.get("competence"),
                "theme": lecon.get("theme"),
                "lecon_titre": titre,
                "habilete": h.get("habilete"),
            })
    return documents


STOPWORDS_META = {
    "de", "la", "le", "les", "des", "du", "a", "en", "et", "dans", "sur",
    "ou", "un", "une", "ce", "cette", "ces", "que", "qui", "pour", "par",
    "sont", "est", "comment", "quelles", "quelle", "quel", "quels",
    "combien", "on", "voit", "voir", "toutes", "tous", "toute", "tout",
    "au", "aux", "avec", "sans", "se", "ses", "son", "sa", "leur", "leurs",
    "plus", "moins", "niveau", "niveaux", "terminale", "classe", "6e",
    "5e", "4e", "3e", "2nde", "1ere", "jusqu", "jusque", "attendues",
    "attendue", "habiletes", "habilete", "lecon", "lecons",
}


def _mots_significatifs(texte: str) -> set:
    mots = re.findall(r"[a-z]+", normaliser(texte))
    return {m.rstrip("s") for m in mots if len(m) >= 4 and m not in STOPWORDS_META}


class MoteurRAG:
    # Formes textuelles possibles pour désigner un niveau dans une question
    # libre ("en 6ème", "au niveau de la 3e", "en terminale C"...), utilisées
    # pour détecter si la question mentionne explicitement un niveau
    # différent de celui sélectionné dans l'interface.
    ALIAS_NIVEAU = {
        "6e": ["6e", "6eme", "6ème", "sixieme", "sixième"],
        "5e": ["5e", "5eme", "5ème", "cinquieme", "cinquième"],
        "4e": ["4e", "4eme", "4ème", "quatrieme", "quatrième"],
        "3e": ["3e", "3eme", "3ème", "troisieme", "troisième"],
        "2A": ["2nde a", "2ndea", "seconde a"],
        "2C": ["2nde c", "2ndec", "seconde c"],
        "1A1": ["1ere a1", "1erea1", "premiere a1"],
        "1A2": ["1ere a2", "1erea2", "premiere a2"],
        "1C": ["1ere c", "1erec", "premiere c"],
        "1D": ["1ere d", "1ered", "premiere d"],
        "TA1": ["terminale a1", "tle a1", "tlea1"],
        "TA2": ["terminale a2", "tle a2", "tlea2"],
        "TC": ["terminale c", "tle c", "tlec"],
        "TD": ["terminale d", "tle d", "tled"],
    }

    # Questions "structurelles" (comptage / liste de leçons) : on répond
    # directement depuis les données, sans jamais passer par le LLM --
    # aucun petit modèle local ne peut "compter" fiablement des éléments
    # qu'il ne voit pas tous en même temps dans son contexte.
    PATRONS_STRUCTURELS = [
        re.compile(r"combien.{0,15}le[çc]on", re.IGNORECASE),
        re.compile(r"nombre de le[çc]on", re.IGNORECASE),
        re.compile(r"(liste|quelles? sont).{0,15}le[çc]on", re.IGNORECASE),
    ]

    PATRONS_GENEALOGIE = [
        re.compile(r"[ée]volu", re.IGNORECASE),          # évolue, évolution
        re.compile(r"[àa] travers les niveaux", re.IGNORECASE),
        re.compile(r"de la 6[èe]?me? [àa] la terminale", re.IGNORECASE),
        re.compile(r"g[ée]n[ée]alogie", re.IGNORECASE),
        re.compile(r"progression.{0,15}(notion|niveaux)", re.IGNORECASE),
    ]

    PATRONS_LECON_COMPLETE = [
        re.compile(r"toutes? les habilet[ée]s?", re.IGNORECASE),
        re.compile(r"contenu (complet|entier) de la le[çc]on", re.IGNORECASE),
        re.compile(r"habilet[ée]s? (attendues?|de la le[çc]on)", re.IGNORECASE),
    ]

    def __init__(self, chemin_json: str):
        with open(chemin_json, encoding="utf-8") as f:
            self.lecons = json.load(f)
        self.documents = construire_documents(self.lecons)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrammes + bigrammes : capte "fonction dérivable", "nombre complexe", etc.
            min_df=1,
            sublinear_tf=True,
        )
        # Index global (sur TOUT le corpus) -- on refait un sous-index à la
        # volée par appel, filtré sur les niveaux autorisés, plutôt que de
        # tout réindexer à chaque fois : plus simple et le corpus est petit
        # (quelques centaines de documents), donc le coût est négligeable.
        self.matrice_globale = self.vectorizer.fit_transform(
            d["texte_norm"] for d in self.documents
        )

    def detecter_niveau_mentionne(self, question: str) -> str:
        """Cherche si la question mentionne explicitement un niveau en toutes
        lettres (ex. "en 6ème"). Retourne le token du niveau si trouvé,
        sinon None."""
        q_norm = normaliser(question)
        for token, alias in self.ALIAS_NIVEAU.items():
            for a in alias:
                if re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", q_norm):
                    return token
        return None

    def repondre_structurelle(self, question: str, niveau_defaut: str):
        """
        Répond directement (sans LLM) aux questions de comptage/liste de
        leçons ("combien de leçons en 6e ?", "liste des leçons de TleC").
        Retourne None si la question ne correspond à aucun de ces patrons
        -- dans ce cas, on repasse par le pipeline recherche + génération
        habituel.
        """
        if not any(p.search(question) for p in self.PATRONS_STRUCTURELS):
            return None

        niveau = self.detecter_niveau_mentionne(question) or niveau_defaut
        lecons_niveau = [l for l in self.lecons if l.get("niveau") == niveau]
        niv_label = NIVEAU_LABELS.get(niveau, niveau)

        if not lecons_niveau:
            return f"Aucune leçon trouvée pour le niveau {niv_label} dans le corpus."

        lignes = [f"En {niv_label}, le programme officiel compte **{len(lecons_niveau)} leçons** :", ""]
        for i, l in enumerate(lecons_niveau, 1):
            lignes.append(f"{i}. {l['lecon_titre']}")
        return "\n".join(lignes)

    def _extraire_titres_mentionnes(self, question: str, niveaux_possibles=None):
        """
        Cherche, parmi les vraies leçons du corpus, celle(s) dont le titre
        se rapporte à la question -- soit par correspondance exacte
        (le titre apparaît tel quel dans la question), soit par
        recoupement de mots-clés significatifs (ex. "fonctions" partagé
        entre la question et le titre "Généralités sur les fonctions").
        Ne déduplique JAMAIS par titre : si plusieurs niveaux partagent le
        même titre de leçon (ex. "Généralités sur les fonctions" en 2ndeC
        ET en 1èreC), les deux doivent être retournés -- c'est justement
        le but de la généalogie.
        """
        candidats = self.lecons
        if niveaux_possibles is not None:
            candidats = [l for l in candidats if l["niveau"] in niveaux_possibles]

        q_norm = normaliser(question)
        mots_question = _mots_significatifs(question)

        resultats = []
        for lecon in candidats:
            titre_norm = normaliser(lecon["lecon_titre"])
            if titre_norm and titre_norm in q_norm:
                resultats.append(lecon)
                continue
            mots_titre = _mots_significatifs(lecon["lecon_titre"])
            if mots_titre:
                # Au moins la moitié des mots significatifs du titre doivent
                # être repris dans la question -- un seul mot générique
                # partagé (ex. "nombres", présent dans "Nombres complexes"
                # ET "Nombres entiers naturels") ne suffit pas à matcher.
                import math
                seuil = math.ceil(len(mots_titre) / 2)
                if len(mots_titre & mots_question) >= seuil:
                    resultats.append(lecon)
        return resultats

    def repondre_lecon_complete(self, question: str, niveau_defaut: str):
        """
        Répond aux questions du type "toutes les habiletés de la leçon X" en
        renvoyant le contenu INTÉGRAL de la leçon (toutes les habiletés,
        tous les contenus), pas seulement les meilleurs fragments trouvés
        par recherche sémantique -- pour garantir zéro oubli.
        """
        if not any(p.search(question) for p in self.PATRONS_LECON_COMPLETE):
            return None

        niveau_mentionne = self.detecter_niveau_mentionne(question)
        autorises = set(niveaux_autorises(niveau_mentionne or niveau_defaut))
        trouvees = self._extraire_titres_mentionnes(question, autorises)

        if not trouvees:
            return (
                "Je n'ai pas identifié de leçon précise dans ta question. "
                "Essaie de citer le titre exact de la leçon (ex. \"toutes les "
                "habiletés de la leçon Nombres complexes\")."
            )

        lignes = []
        for lecon in trouvees:
            niv_label = NIVEAU_LABELS.get(lecon["niveau"], lecon["niveau"])
            lignes.append(f"## [{niv_label}] {lecon['lecon_titre']}\n")
            for h in lecon.get("habiletes", []):
                lignes.append(f"**{h['habilete']}**")
                for c in h.get("contenus", []):
                    lignes.append(f"- {c}")
                lignes.append("")
        return "\n".join(lignes)

    def repondre_genealogie(self, question: str, niveau_defaut: str):
        """
        Répond aux questions sur l'évolution d'une notion à travers les
        niveaux ("comment évoluent les fonctions de la 6e à la terminale ?").
        Cherche TOUTES les leçons dont le titre correspond, sur TOUS les
        niveaux (pas seulement ceux autorisés pour un niveau donné -- ici on
        veut voir toute la progression, y compris au-delà du niveau courant).
        """
        if not any(p.search(question) for p in self.PATRONS_GENEALOGIE):
            return None

        trouvees = self._extraire_titres_mentionnes(question, niveaux_possibles=None)
        if not trouvees:
            return (
                "Je n'ai pas identifié de notion précise dans ta question. "
                "Essaie de citer le nom de la leçon ou de la notion "
                "(ex. \"comment évoluent les fonctions de la 6e à la terminale ?\")."
            )

        trouvees_triees = sorted(
            trouvees, key=lambda l: ANNEE_RANG.get(l["niveau"], 99)
        )

        lignes = [f"**Progression trouvée à travers {len(trouvees_triees)} niveau(x) :**\n"]
        for lecon in trouvees_triees:
            niv_label = NIVEAU_LABELS.get(lecon["niveau"], lecon["niveau"])
            nb_hab = len(lecon.get("habiletes", []))
            lignes.append(f"### {niv_label} — {lecon['lecon_titre']} ({nb_hab} habileté(s))")
            for h in lecon.get("habiletes", [])[:4]:  # aperçu, pas la liste intégrale ici
                lignes.append(f"- **{h['habilete']}** : {'; '.join(h.get('contenus', [])[:2])}")
            lignes.append("")
        return "\n".join(lignes)

    def _doc_pour_habilete(self, lecon: dict, h: dict) -> dict:
        contenus = " ; ".join(h.get("contenus", []))
        return {
            "texte": f"{h['habilete']} : {contenus}",
            "niveau": lecon.get("niveau"),
            "competence": lecon.get("competence"),
            "theme": lecon.get("theme"),
            "lecon_titre": lecon.get("lecon_titre"),
            "habilete": h.get("habilete"),
        }

    def rechercher(self, question: str, niveau_cible: str, k: int = 5) -> list:
        autorises = set(niveaux_autorises(niveau_cible))

        # 1. Essai prioritaire : la question nomme-t-elle clairement UNE
        # leçon précise, sans ambiguïté ? Si oui, on renvoie TOUT son
        # contenu (garanti complet et exact) plutôt que de risquer que le
        # TF-IDF se trompe carrément de leçon -- c'est le bug qu'on vient
        # de voir sur "les nombres réels en 2ndeC" (retombé sur "Fonctions"
        # au lieu de "Ensemble des nombres réels").
        lecons_nommees = self._extraire_titres_mentionnes(question, niveaux_possibles=autorises)
        if len(lecons_nommees) == 1:
            lecon = lecons_nommees[0]
            return [
                {**self._doc_pour_habilete(lecon, h), "score": 1.0}
                for h in lecon.get("habiletes", [])
            ]

        # 2. Sinon (aucune leçon reconnue nommément, ou plusieurs candidates
        # ambiguës), recherche floue habituelle.
        indices_filtres = [
            i for i, d in enumerate(self.documents) if d["niveau"] in autorises
        ]
        if not indices_filtres:
            return []

        # Verbes d'habileté détectés dans la question (ex. "Résous" ->
        # stem "reso", à rapprocher de "Résoudre"). Utilisé pour le bonus,
        # séparément du score de contenu.
        mots_question = re.findall(r"[a-zàâäéèêëïîôöùûüç]+", normaliser(question))
        stems_question = {_stem_verbe(m) for m in mots_question if len(m) >= 4}
        stems_verbes_detectes = stems_question & STEMS_HABILETE

        vecteur_question = self.vectorizer.transform([normaliser(question)])
        sous_matrice = self.matrice_globale[indices_filtres]
        scores_contenu = cosine_similarity(vecteur_question, sous_matrice)[0]

        scores_combines = []
        for pos, idx in enumerate(indices_filtres):
            score = float(scores_contenu[pos])
            if stems_verbes_detectes:
                stem_habilete = _stem_verbe(self.documents[idx]["habilete"])
                if stem_habilete in stems_verbes_detectes:
                    score += BONUS_VERBE
            scores_combines.append((idx, score))

        classement = sorted(scores_combines, key=lambda x: x[1], reverse=True)[:k]

        resultats = []
        for idx, score in classement:
            if score <= 0:
                continue
            d = self.documents[idx]
            resultats.append({**d, "score": round(score, 3)})
        return resultats

    def contexte_pour_prompt(self, resultats: list) -> str:
        lignes = []
        for r in resultats:
            niv = NIVEAU_LABELS.get(r["niveau"], r["niveau"])
            lignes.append(
                f"[{niv} | {r['lecon_titre']}] {r['habilete']} : "
                + r["texte"].split(" : ", 1)[-1]
            )
        return "\n".join(lignes)


_generator = None
_generator_nom = None

# Deux façons de générer une réponse, au choix :
#   - "local"  : transformers + modèle en cache HF (ex. Qwen2.5-0.5B-Instruct)
#                -> rapide à charger, aucune dépendance réseau, mais petit modèle.
#   - "ollama" : appel HTTP à un modèle servi par Ollama (ex. mistral:7b-instruct-q4_K_M)
#                -> Ollama doit tourner (icône barre système), modèle plus gros donc
#                   potentiellement meilleure qualité, mais plus lent en CPU pur.
MODELE_LOCAL_PAR_DEFAUT = "Qwen/Qwen2.5-0.5B-Instruct"
MODELE_OLLAMA_PAR_DEFAUT = "qwen2.5:3b-instruct-q4_K_M"


def _get_generator(nom_modele: str = MODELE_LOCAL_PAR_DEFAUT):
    """Charge le pipeline transformers une seule fois par modèle (paresseux)."""
    global _generator, _generator_nom
    if _generator is None or _generator_nom != nom_modele:
        from transformers import pipeline
        _generator = pipeline("text-generation", model=nom_modele)
        _generator_nom = nom_modele
    return _generator


def construire_messages(question: str, contexte: str, historique: list = None) -> list:
    """
    Construit la liste de messages (format chat) envoyée au modèle.

    `historique` : liste de dicts {"role": "user"|"assistant", "content": ...}
    des échanges précédents de la conversation. On ne garde que les 2
    derniers échanges (4 messages) pour ne pas saturer le contexte d'un
    petit modèle local -- au-delà, la qualité se dégrade plus qu'elle ne
    s'améliore sur ce genre de modèle.
    """
    system = (
        "Tu es un assistant pédagogique pour des professeurs de mathématiques "
        "en Côte d'Ivoire. Réponds UNIQUEMENT à partir du contexte du programme "
        "officiel fourni à chaque question. Si le contexte ne permet pas de "
        "répondre précisément, dis-le clairement plutôt que d'inventer."
    )
    messages = [{"role": "system", "content": system}]
    if historique:
        messages.extend(historique[-4:])
    messages.append({
        "role": "user",
        "content": f"CONTEXTE DU PROGRAMME OFFICIEL :\n{contexte}\n\nQUESTION : {question}",
    })
    return messages


def _generer_local(messages: list, max_new_tokens: int, nom_modele: str) -> str:
    generator = _get_generator(nom_modele)
    sortie = generator(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=True,
    )
    return sortie[0]["generated_text"][-1]["content"].strip()


def _generer_ollama(messages: list, max_new_tokens: int, nom_modele: str) -> str:
    import requests

    try:
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": nom_modele,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": max_new_tokens},
            },
            timeout=900,  # 15 min : Mistral 7B en CPU pur peut être très lent
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Impossible de joindre Ollama sur http://localhost:11434. "
            "Vérifie que l'application Ollama tourne bien en arrière-plan."
        )
    except requests.exceptions.ReadTimeout:
        return (
            f"⚠️ Le modèle {nom_modele} a mis plus de 15 minutes à répondre "
            "(timeout). Essaie un modèle plus léger (Qwen2.5-0.5B local, ou "
            "Qwen2.5-3B) ou reformule une question plus courte."
        )


def generer_reponse(
    question: str,
    contexte: str,
    historique: list = None,
    max_new_tokens: int = 400,
    moteur_generation: str = "local",
    nom_modele: str = None,
) -> str:
    """
    Génère une réponse à partir du contexte RAG.

    `moteur_generation` :
        "local"  -> transformers, modèle par défaut Qwen2.5-0.5B-Instruct
                    (rapide, aucune dépendance réseau, mais petit modèle)
        "ollama" -> Ollama doit tourner ; modèle par défaut
                    mistral:7b-instruct-q4_K_M (plus gros, plus lent)

    `nom_modele` : surcharge le modèle par défaut du moteur choisi (ex. passer
    "qwen2.5:3b-instruct-q4_K_M" avec moteur_generation="ollama" une fois ce
    modèle téléchargé).
    """
    messages = construire_messages(question, contexte, historique)

    if moteur_generation == "ollama":
        modele = nom_modele or MODELE_OLLAMA_PAR_DEFAUT
        return _generer_ollama(messages, max_new_tokens, modele)
    elif moteur_generation == "local":
        modele = nom_modele or MODELE_LOCAL_PAR_DEFAUT
        return _generer_local(messages, max_new_tokens, modele)
    else:
        raise ValueError(
            f"moteur_generation inconnu : {moteur_generation!r}. "
            "Utilise 'local' ou 'ollama'."
        )


if __name__ == "__main__":
    import sys

    chemin_json = sys.argv[1] if len(sys.argv) > 1 else "resultats_pdf_2023.json"
    moteur = MoteurRAG(chemin_json)

    questions_test = [
        ("Comment dériver une fonction composée ?", "TC"),
        ("Comment résoudre une équation différentielle du type f'+af=0 ?", "TD"),
        ("Comment calculer le PGCD de deux nombres ?", "3e"),
    ]

    for question, niveau in questions_test:
        print(f"\n{'='*70}\nQuestion ({niveau}) : {question}\n{'='*70}")
        resultats = moteur.rechercher(question, niveau, k=5)
        if not resultats:
            print("Aucun résultat pertinent trouvé dans le périmètre autorisé.")
            continue
        for r in resultats:
            niv = NIVEAU_LABELS.get(r["niveau"], r["niveau"])
            print(f"  [{r['score']}] ({niv} | {r['lecon_titre']}) {r['texte']}")

    # ------------------------------------------------------------------
    # Test bout-en-bout : recherche + génération, sur une seule question,
    # pour valider toute la chaîne (retire ce bloc si tu veux juste tester
    # la recherche seule, plus rapide, sans charger le modèle).
    # ------------------------------------------------------------------
    print(f"\n{'='*70}\nTest complet (recherche + génération)\n{'='*70}")
    question_finale = "Comment résoudre une équation différentielle du type f'+af=0 ?"
    niveau_finale = "TD"
    resultats_finale = moteur.rechercher(question_finale, niveau_finale, k=3)
    contexte = moteur.contexte_pour_prompt(resultats_finale)
    print("--- Contexte injecté dans le prompt ---")
    print(contexte)
    print("\n--- Génération du modèle (Qwen2.5-0.5B, patience, premier chargement plus lent) ---")
    reponse = generer_reponse(question_finale, contexte)
    print(reponse)

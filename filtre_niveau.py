r"""
Outil de filtrage "par niveau et en dessous" pour le RAG Maths CIV.

Idée : le système ivoirien n'a PAS un seul ordre linéaire de niveaux.
- De la 6e à la 3e : programme commun, ordre linéaire simple.
- À partir de la 2nde : le cursus se divise par série. Un élève de TleC
  n'a jamais suivi le programme de TleD -- il faut donc remonter la
  bonne filière (TleC -> 1èreC -> 2ndeC -> 3e -> ... -> 6e), pas mélanger
  toutes les séries entre elles.

Ce module ne fait AUCUNE recherche sémantique : il calcule juste, pour un
niveau cible donné, l'ensemble exact des niveaux dont on a le droit
d'utiliser les leçons/habiletés (le niveau cible + tous ses "ancêtres"
dans sa filière). Ce résultat sert ensuite de filtre STRICT avant toute
recherche RAG -- le modèle de génération ne voit jamais de contenu hors
de cet ensemble.

Filiation et passerelles confirmées avec l'utilisateur (système éducatif
ivoirien) : collège commun 6e->5e->4e->3e (BEPC), puis 2ndeA->1èreA1/1èreA2,
2ndeC->1èreC/1èreD, chaque série de 1ère menant directement à sa Terminale.
"""

# Pour chaque niveau, quel est son "niveau parent" direct dans la
# progression (le niveau immédiatement en dessous, dans la même filière).
# Racine (6e) -> pas de parent (None).
FILIATION = {
    "6e": None,
    "5e": "6e",
    "4e": "5e",
    "3e": "4e",
    # Collège commun -> obtention du BEPC -> Seconde se divise en 2 séries
    "2A": "3e",
    "2C": "3e",
    # 2ndeA se divise en 1èreA1 / 1èreA2 (séries littéraires ; A1 plus orientée maths que A2)
    "1A1": "2A",
    "1A2": "2A",
    # 2ndeC se divise en 1èreC / 1èreD (séries scientifiques)
    "1C": "2C",
    "1D": "2C",
    # Chaque série de 1ère continue directement en Terminale (parcours standard)
    "TA1": "1A1",
    "TA2": "1A2",
    "TC": "1C",
    "TD": "1D",
}

# Rang de rigueur mathématique transversal entre séries, du plus faible au plus
# élevé, confirmé par l'utilisateur (système éducatif ivoirien) : A2 < A1 < D < C.
# NB : ce classement n'est PAS utilisé par niveaux_autorises() ci-dessous -- il
# sert uniquement de métadonnée informative (voir PASSERELLES_OFFICIELLES).
SERIE_RANG = {"A2": 0, "A1": 1, "D": 2, "C": 3}

# Passerelles officielles de réorientation en fin de 1ère (informationnel).
# Un élève de 1ère peut rejoindre n'importe quelle Terminale de rang <= son
# propre rang (ex. 1èreC, rang 3, peut rejoindre TleD/TleA1/TleA2/TleC ;
# 1èreD, rang 2, peut rejoindre TleA1/TleA2/TleD mais PAS TleC).
# ATTENTION : ceci ne veut PAS dire qu'un élève de TleD a forcément suivi le
# contenu de 1èreC -- il a pu suivre le parcours standard (1èreD). Pour rester
# prudent sur ce qu'on peut garantir comme "déjà vu", niveaux_autorises()
# n'utilise QUE le parcours standard (FILIATION), jamais ces passerelles.
PASSERELLES_OFFICIELLES = {
    "1A2": ["TA2"],
    "1A1": ["TA1", "TA2"],
    "1D":  ["TD", "TA1", "TA2"],
    "1C":  ["TC", "TD", "TA1", "TA2"],
}


NIVEAU_LABELS = {
    "6e": "6ème", "5e": "5ème", "4e": "4ème", "3e": "3ème",
    "2A": "2ndeA", "2C": "2ndeC",
    "1A1": "1èreA1", "1A2": "1èreA2", "1C": "1èreC", "1D": "1èreD",
    "TA1": "TleA1", "TA2": "TleA2", "TC": "TleC", "TD": "TleD",
}


def niveaux_autorises(niveau_cible: str) -> list:
    """
    Retourne, dans l'ordre du plus avancé au plus élémentaire, la liste des
    niveaux dont le contenu est valide pour un élève de `niveau_cible`
    (le niveau cible lui-même + tous ses niveaux "ancêtres" dans sa filière).

    Lève une erreur claire si le niveau n'est pas reconnu (plutôt que de
    filtrer silencieusement sur un ensemble vide, ce qui masquerait le bug).
    """
    if niveau_cible not in FILIATION:
        raise ValueError(
            f"Niveau inconnu : {niveau_cible!r}. "
            f"Niveaux valides : {sorted(FILIATION.keys())}"
        )

    chaine = []
    courant = niveau_cible
    while courant is not None:
        chaine.append(courant)
        courant = FILIATION[courant]
    return chaine


def filtrer_lecons(lecons: list, niveau_cible: str) -> list:
    """
    `lecons` : liste de dicts au format produit par run_all_pdfs.py /
    pdf_parser.py (chaque leçon a une clé "niveau").

    Retourne uniquement les leçons dont le niveau est autorisé pour
    `niveau_cible` (le niveau cible et ses ancêtres de filière).
    """
    autorises = set(niveaux_autorises(niveau_cible))
    return [l for l in lecons if l.get("niveau") in autorises]


def contexte_pour_prompt(lecons_filtrees: list, niveau_cible: str, max_lecons: int = None) -> str:
    """
    Construit un bloc de texte prêt à être injecté dans le prompt du modèle
    de génération : la liste des leçons/habiletés/contenus autorisés,
    avec le niveau d'origine de chacune (utile pour que le modèle explique
    d'où vient une habileté, ex. "cette notion a été vue en 4e").
    """
    lignes = [
        f"Contexte pédagogique autorisé pour le niveau {NIVEAU_LABELS.get(niveau_cible, niveau_cible)} "
        f"et les niveaux antérieurs de sa filière :",
        "",
    ]
    a_afficher = lecons_filtrees[:max_lecons] if max_lecons else lecons_filtrees
    for l in a_afficher:
        niv_label = NIVEAU_LABELS.get(l.get("niveau"), l.get("niveau"))
        lignes.append(f"[{niv_label}] {l.get('lecon_titre')}")
        for h in l.get("habiletes", []):
            lignes.append(f"  - {h['habilete']} : " + " ; ".join(h["contenus"][:3]))
    return "\n".join(lignes)


if __name__ == "__main__":
    # Petit jeu de données factice pour vérifier la logique de filtrage
    # sans dépendre du vrai resultats_pdf_2023.json.
    lecons_test = [
        {"niveau": "6e", "lecon_titre": "Nombres entiers naturels", "habiletes": []},
        {"niveau": "5e", "lecon_titre": "Nombres premiers", "habiletes": []},
        {"niveau": "4e", "lecon_titre": "Calcul littéral", "habiletes": []},
        {"niveau": "3e", "lecon_titre": "Racines carrées", "habiletes": []},
        {"niveau": "2A", "lecon_titre": "Calcul numérique (2ndeA)", "habiletes": []},
        {"niveau": "2C", "lecon_titre": "Ensemble des nombres réels (2ndeC)", "habiletes": []},
        {"niveau": "1C", "lecon_titre": "Limites et continuité (1èreC)", "habiletes": []},
        {"niveau": "1D", "lecon_titre": "Limites et continuité (1èreD)", "habiletes": []},
        {"niveau": "TC", "lecon_titre": "Nombres complexes (TleC)", "habiletes": []},
        {"niveau": "TD", "lecon_titre": "Nombres complexes (TleD)", "habiletes": []},
    ]

    for cible in ["3e", "TC", "1D", "TA2"]:
        print(f"\n=== Niveaux autorisés pour {cible} : {niveaux_autorises(cible)} ===")
        for l in filtrer_lecons(lecons_test, cible):
            print(f"   [{l['niveau']}] {l['lecon_titre']}")

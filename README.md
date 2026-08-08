# 📐 Assistant Maths CIV

Un assistant pour **apprendre et maîtriser le programme officiel de mathématiques de Côte d'Ivoire**, de la 6ème à la Terminale, toutes séries confondues (A1, A2, C, D).

👉 **[Utiliser l'assistant en ligne](#)** *(lien à ajouter une fois déployé sur Streamlit Cloud)*

---

## 🎯 À quoi ça sert ?

Le programme officiel de mathématiques (DPFC, Côte d'Ivoire) fait plusieurs centaines de pages, réparties en 14 documents différents selon le niveau et la série. Retrouver précisément **quelle notion est enseignée où**, **quelles habiletés sont attendues sur une leçon donnée**, ou **comment une notion évolue d'un niveau à l'autre** demande normalement de fouiller à la main dans ces documents.

Cet assistant permet de poser une question en langage naturel et de retrouver **instantanément et fidèlement** le passage exact du programme officiel qui y répond — sans reformulation, sans risque d'erreur ou d'invention : tu vois toujours le texte réel du programme.

C'est pensé en priorité pour :
- les **élèves** qui veulent réviser ou vérifier une notion à leur niveau,
- les **enseignants** qui préparent une leçon ou veulent situer une notion dans la progression globale,
- toute personne qui veut comprendre **comment les mathématiques s'enchaînent, de la 6ème à la Terminale**, à travers les différentes séries.

## ✨ Fonctionnalités

- **Recherche par niveau et par filière** — les réponses ne s'appuient que sur le programme du niveau choisi et des niveaux qui le précèdent réellement dans son propre parcours (le système ivoirien se divise en filières à partir de la Seconde : A1, A2, C, D — l'assistant respecte cette logique plutôt que de tout mélanger).
- **Contenu complet d'une leçon** — demande "toutes les habiletés de la leçon Nombres complexes" (par exemple) pour obtenir l'intégralité du contenu officiel, sans rien manquer.
- **Généalogie d'une notion** — demande "comment évoluent les fonctions de la 6e à la terminale ?" pour voir, niveau par niveau, comment une notion se construit au fil de la scolarité.
- **Comptage et listes de leçons** — "combien de leçons y a-t-il en 3e ?", "quelles sont les leçons de Terminale C ?"

## 📖 Mode consultation

Cette version publique fonctionne en **mode consultation** : les réponses affichent directement le contenu exact du programme officiel trouvé, sans reformulation par un modèle d'intelligence artificielle. C'est un choix délibéré : ça garantit une fidélité totale au texte officiel, sans aucun risque d'erreur ou d'invention, et ça fonctionne pour un nombre illimité de personnes en même temps, gratuitement.

## 🗂️ Source des données

Le contenu provient des **Programmes Éducatifs et Guides d'Exécution — Mathématiques** publiés par la Direction de la Pédagogie et de la Formation Continue (DPFC) de Côte d'Ivoire (édition 2023), couvrant les 14 niveaux/séries de la 6ème à la Terminale.

## 🚀 Utilisation

1. Choisis ton niveau dans le menu déroulant (au-dessus de la zone de question).
2. Pose ta question en français, comme tu la formulerais naturellement (ex. *"comment résoudre une équation différentielle f'+af=0 ?"*, *"les nombres réels en 2ndeC portent sur quelles notions ?"*).
3. Consulte le contenu officiel trouvé, organisé par leçon.

## 🛠️ Faire tourner le projet en local

```bash
git clone https://github.com/GMeless/programme-maths-civ.git
cd programme-maths-civ
pip install -r requirements.txt
streamlit run programme_maths_civ.py
```

## 🙏 Remarque

Ce projet est un outil d'aide à la révision et à la préparation pédagogique. Il ne remplace pas la lecture du programme officiel complet ni l'encadrement d'un enseignant. En cas de doute sur une notion, réfère-toi toujours au document officiel de la DPFC.

---

*Projet réalisé par [@GMeless](https://github.com/GMeless).*

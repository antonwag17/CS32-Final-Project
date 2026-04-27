# Career-Match

A career recommender for college students. The student answers a 20-question survey; the tool maps their answers into a 10-dimensional feature vector, assigns them to the nearest career archetype, and identifies the single best-matching career within that archetype.

## How it works

1. Student answers 20 questions rated 0–5
2. Answers are converted into a 10-dim vector (analytical, creative, people-facing, independent, risk tolerance, prestige, structured, impact, technical depth, entrepreneurial)
3. The vector is compared against 10 archetype centroids (computed by averaging the career vectors in each group)
4. The student is assigned to the nearest archetype
5. Within that archetype, the closest individual career is found and displayed

## Setup

```bash
pip install -r requirements.txt
python3 app.py
```

## Project structure

```
app.py                  — main CLI application
questionnaire.py        — 20-question survey and answer-to-vector mapping
categorize_careers.py   — hardcoded career vectors, archetype definitions, centroid computation
requirements.txt
```

## Archetypes

| # | Archetype |
|---|---|
| 0 | Consulting |
| 1 | Finance |
| 2 | Technology |
| 3 | Academia / Research |
| 4 | Engineering |
| 5 | Public Service / Nonprofit |
| 6 | Arts / Entertainment |
| 7 | Government / Politics |
| 8 | Entrepreneurship |
| 9 | Law |

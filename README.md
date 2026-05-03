# Career-Match

A career recommender for college students. The student answers a 20-question survey; the app maps their answers into a 10-dimensional feature vector, assigns them to the nearest career archetype, and surfaces the top matching careers with an interactive radar chart of their trait profile.

## How it works

1. Student rates 20 statements from 1 (strongly disagree) to 5 (strongly agree)
2. Answers are averaged into a 10-dim vector across these dimensions:
   - Analytical Thinking, Creativity, People & Collaboration, Independence
   - Risk Tolerance, Prestige & Recognition, Structure & Process, Impact & Purpose
   - Technical Depth, Entrepreneurship
3. The vector is compared against 10 archetype centroids (computed by averaging the career vectors in each group)
4. If the result is ambiguous (near-tie, flat profile, or conflicting signals), a short set of targeted follow-up questions refines the match
5. The student is assigned to the nearest archetype and shown their top 3 career matches with match percentages

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py      — main web application (Streamlit)
questionnaire.py      — 20-question survey, feature definitions, answer-to-vector mapping
categorize_careers.py — hardcoded career vectors, archetype definitions, centroid computation
requirements.txt      — Python dependencies (numpy, streamlit, plotly)
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

## AI tools

This project used Claude Code. Specifically:

- **UI/UX design**: Claude wrote the full Streamlit front-end, including all CSS styling, the radar chart, and the session-state flow between the survey, edge-case refinement, and results screens.
- **Code review and cleanup**: Claude audited the codebase for dead code, duplicate definitions, and unused imports, and removed them.

# CS32 Final Project — Student-to-Profession Matching Algorithm

A Python-based algorithm that matches students to professions and roles based on their interests, personal attributes, and academic performance metrics.

---

## Overview

This project implements a matching system that takes student profile data as input and recommends suitable professions or career roles. The algorithm analyzes multiple dimensions of a student's profile — including self-reported interests, personality/attribute scores, and measurable academic performance — to produce a ranked list of career matches.

---

## Features

- Collects and processes student interest data across multiple domains
- Incorporates personal attribute weightings (e.g., creativity, analytical thinking, leadership)
- Uses academic performance metrics to refine match quality
- Produces a ranked list of profession/role recommendations per student
- Modular design for easy extension with new professions or scoring criteria

---

## Algorithm Design

The matching algorithm works in three stages:

1. **Profile Building** — Student data (interests, attributes, GPA/grades by subject) is collected and normalized into a standard profile vector.
2. **Scoring** — Each profession in the database is scored against the student's profile using a weighted similarity function. Weights reflect the relative importance of interests, attributes, and performance for each profession.
3. **Ranking** — Professions are sorted by their match score and returned as an ordered list of recommendations.

---

## Project Structure

```
CS32-Final-Project/
├── README.md
└── ...                  # source files (to be added)
```

---

## Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
git clone https://github.com/antonwag17/CS32-Final-Project.git
cd CS32-Final-Project
```

### Usage

```bash
python main.py
```

---

## Input Format

Student profiles are expected to include the following fields:

| Field | Type | Description |
|---|---|---|
| `interests` | list of str | Subject/domain areas the student is interested in |
| `attributes` | dict | Named attribute scores (e.g., `{"creativity": 8, "leadership": 6}`) |
| `grades` | dict | Subject-level performance scores (e.g., `{"math": 92, "english": 85}`) |

---

## Output Format

The algorithm returns a ranked list of profession matches:

```python
[
    {"profession": "Software Engineer", "score": 0.91},
    {"profession": "Data Analyst",      "score": 0.87},
    {"profession": "UX Designer",       "score": 0.74},
    ...
]
```

---

## Authors

- Anton Wagner — [antonwag17](https://github.com/antonwag17)

---

## Course

CS32 — Final Project

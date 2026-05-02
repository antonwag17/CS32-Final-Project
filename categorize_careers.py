"""
Hardcoded Harvard-relevant career vectors and archetype definitions.

Each career is scored 1-5 on 10 dimensions matching the student questionnaire.
Archetype centroids are computed at import time by averaging career vectors.
"""

import numpy as np

FEATURES = [
    "analytical", "creative", "people_facing", "independent",
    "risk_tolerance", "prestige", "structured", "impact",
    "technical_depth", "entrepreneurial",
]

ARCHETYPE_NAMES = {
    0: "Consulting",
    1: "Finance",
    2: "Technology",
    3: "Academia / Research",
    4: "Engineering",
    5: "Public Service / Nonprofit",
    6: "Arts / Entertainment",
    7: "Government / Politics",
    8: "Entrepreneurship",
    9: "Law",
}

ARCHETYPE_DESCRIPTIONS = {
    0: "Fast-paced client work, cross-industry problem-solving, and career prestige.",
    1: "Data-driven, high-stakes financial markets and top-tier compensation.",
    2: "Building software products, systems, and data solutions at scale.",
    3: "Deep intellectual inquiry, publishing research, and advancing human knowledge.",
    4: "Designing and building physical and technical systems that shape the world.",
    5: "Improving lives through healthcare, advocacy, and community work.",
    6: "Expressing ideas through writing, visual media, design, or performance.",
    7: "Shaping policy, diplomacy, and the institutions that govern society.",
    8: "Creating new ventures, taking calculated risks, and building from scratch.",
    9: "Navigating legal systems to advise, advocate, and uphold justice.",
}

# (title, cluster_id, [analytical, creative, people_facing, independent,
#                      risk_tolerance, prestige, structured, impact, technical_depth, entrepreneurial])
# Scores are 1-5.
CAREERS: list[tuple[str, int, list[int]]] = [
    # Consulting (0)
    ("Management Consultant",    0, [4, 3, 4, 2, 3, 4, 3, 3, 3, 3]),
    ("Strategy Consultant",      0, [4, 3, 4, 2, 3, 4, 3, 3, 3, 3]),
    ("Operations Consultant",    0, [4, 2, 4, 2, 3, 3, 4, 3, 3, 3]),
    ("Business Analyst",         0, [4, 2, 3, 3, 2, 3, 4, 2, 3, 2]),
    ("Market Research Analyst",  0, [4, 2, 3, 3, 2, 3, 4, 2, 3, 2]),
    ("Human Capital Consultant", 0, [3, 3, 5, 2, 2, 3, 3, 3, 2, 2]),
    # Finance (1)
    ("Investment Banking Analyst", 1, [5, 2, 3, 2, 4, 5, 4, 2, 4, 3]),
    ("Private Equity Associate",   1, [5, 2, 3, 2, 4, 5, 3, 2, 4, 4]),
    ("Hedge Fund Analyst",         1, [5, 2, 2, 3, 5, 5, 3, 1, 4, 3]),
    ("Portfolio Manager",          1, [5, 2, 3, 3, 4, 4, 3, 2, 4, 3]),
    ("Venture Capitalist (Finance)", 1, [4, 3, 4, 3, 5, 5, 2, 3, 3, 5]),
    ("Financial Analyst",          1, [4, 2, 3, 3, 3, 4, 4, 2, 3, 2]),
    # Technology (2)
    ("Software Engineer",         2, [5, 3, 3, 4, 3, 4, 3, 3, 5, 3]),
    ("Technical Product Manager",  2, [4, 4, 4, 3, 3, 4, 3, 3, 3, 4]),
    ("Data Scientist",            2, [5, 3, 2, 4, 2, 4, 3, 3, 5, 2]),
    ("Machine Learning Engineer", 2, [5, 3, 2, 4, 3, 4, 3, 3, 5, 3]),
    ("Technical Program Manager", 2, [4, 3, 4, 2, 3, 3, 4, 3, 4, 3]),
    ("Startup Founder",           2, [4, 4, 4, 4, 5, 4, 2, 4, 4, 5]),
    # Academia / Research (3)
    ("University Professor",    3, [4, 4, 4, 4, 2, 4, 3, 4, 4, 2]),
    ("Research Scientist",      3, [5, 4, 2, 4, 2, 3, 3, 4, 5, 2]),
    ("Postdoctoral Researcher", 3, [5, 3, 2, 4, 2, 3, 3, 3, 5, 1]),
    ("Think Tank Analyst",      3, [4, 3, 3, 4, 2, 3, 3, 4, 3, 2]),
    ("Policy Researcher",       3, [4, 3, 3, 4, 2, 3, 3, 4, 3, 2]),
    ("Lab Director",            3, [4, 3, 4, 4, 2, 4, 3, 4, 4, 3]),
    # Engineering (4)
    ("Civil Engineer",         4, [4, 3, 3, 3, 2, 3, 4, 3, 5, 2]),
    ("Biomedical Engineer",    4, [5, 3, 3, 3, 3, 3, 4, 4, 5, 2]),
    ("Aerospace Engineer",     4, [5, 3, 3, 3, 3, 4, 4, 3, 5, 2]),
    ("Environmental Engineer", 4, [4, 3, 3, 3, 2, 3, 4, 5, 4, 2]),
    ("Electrical Engineer",    4, [5, 3, 2, 3, 2, 3, 4, 3, 5, 2]),
    ("Chemical Engineer",      4, [5, 3, 2, 3, 3, 3, 4, 3, 5, 2]),
    # Public Service / Nonprofit (5)
    ("Nonprofit Director",       5, [3, 3, 5, 3, 3, 3, 3, 5, 2, 4]),
    ("Public Health Officer",    5, [4, 3, 4, 3, 2, 3, 4, 5, 3, 2]),
    ("Physician / Doctor",       5, [4, 3, 4, 3, 3, 5, 4, 5, 4, 2]),
    ("NGO Program Manager",      5, [3, 3, 4, 3, 3, 2, 3, 5, 2, 3]),
    ("Community Organizer",      5, [3, 3, 5, 3, 3, 2, 2, 5, 2, 3]),
    ("International Aid Worker", 5, [3, 3, 4, 3, 4, 2, 3, 5, 2, 3]),
    # Arts / Entertainment (6)
    ("Journalist",       6, [3, 4, 4, 4, 3, 3, 2, 4, 2, 3]),
    ("Author / Writer",  6, [3, 5, 2, 5, 4, 3, 2, 3, 2, 3]),
    ("Film Director",    6, [3, 5, 4, 4, 4, 4, 2, 3, 3, 4]),
    ("Architect",        6, [4, 5, 3, 3, 3, 3, 3, 3, 4, 3]),
    ("Museum Curator",   6, [4, 4, 3, 4, 1, 3, 3, 3, 3, 2]),
    ("Graphic Designer", 6, [3, 5, 3, 4, 2, 2, 2, 2, 3, 3]),
    # Government / Politics (7)
    ("Foreign Service Officer",    7, [4, 3, 4, 3, 3, 4, 3, 4, 3, 2]),
    ("Policy Advisor",             7, [4, 3, 4, 3, 2, 4, 3, 4, 3, 2]),
    ("Legislative Aide",           7, [3, 3, 4, 2, 2, 3, 3, 4, 2, 2]),
    ("Intelligence Analyst",       7, [5, 3, 2, 3, 3, 4, 4, 4, 4, 2]),
    ("Diplomat",                   7, [4, 3, 5, 3, 3, 4, 3, 4, 2, 2]),
    ("Political Campaign Manager", 7, [4, 4, 5, 3, 4, 4, 3, 4, 2, 4]),
    # Entrepreneurship (8)
    ("Startup Founder / CEO",        8, [4, 4, 4, 5, 5, 5, 2, 4, 3, 5]),
    ("Venture Capitalist (Startup)",  8, [4, 3, 4, 3, 5, 5, 2, 3, 3, 5]),
    ("Business Development Manager", 8, [3, 3, 4, 3, 3, 3, 3, 3, 2, 4]),
    ("General Manager",              8, [4, 3, 4, 3, 3, 4, 3, 3, 3, 3]),
    ("Social Entrepreneur",          8, [3, 4, 4, 4, 4, 3, 2, 5, 2, 5]),
    ("Growth Product Manager",       8, [4, 4, 4, 3, 3, 4, 3, 3, 3, 4]),
    # Law (9)
    ("Corporate Attorney",           9, [4, 3, 4, 2, 3, 5, 4, 2, 4, 2]),
    ("Public Defender",              9, [4, 3, 4, 3, 3, 3, 4, 5, 3, 2]),
    ("Federal Judge",                9, [5, 2, 4, 4, 2, 5, 5, 4, 4, 1]),
    ("District Attorney",            9, [4, 3, 4, 3, 3, 4, 4, 4, 3, 2]),
    ("International Law Specialist", 9, [4, 3, 4, 3, 3, 4, 4, 4, 4, 2]),
    ("Legal Counsel",                9, [4, 3, 4, 3, 3, 4, 4, 3, 4, 2]),
]


def _compute_centroids() -> np.ndarray:
    n = len(ARCHETYPE_NAMES)
    totals = np.zeros((n, len(FEATURES)))
    counts = np.zeros(n)
    for _, cluster_id, scores in CAREERS:
        totals[cluster_id] += scores
        counts[cluster_id] += 1
    return totals / counts[:, np.newaxis]


CENTROIDS = _compute_centroids()

"""
Student questionnaire mapping to a 10-dim career feature space.

Each question is rated 0-5. Answers are aggregated via a weighted average
per feature and linearly mapped to the 1-5 scale used by career vectors.
"""

import numpy as np

# The 10 career dimensions. Order here defines the index used throughout.
# NOTE: this list is also defined in categorize_careers.py — both must stay in sync.
FEATURES = [
    "analytical", "creative", "people_facing", "independent",
    "risk_tolerance", "prestige", "structured", "impact",
    "technical_depth", "entrepreneurial",
]

# Each entry is (question text, weight dict).
# Questions map entirely to one dimension (weight 1.0).
QUESTIONS: list[tuple[str, dict[str, float]]] = [
    ("I enjoy analyzing data, building models, or solving quantitative problems.",
     {"analytical": 1.0}),
    ("Breaking down complex problems into logical steps is something I genuinely enjoy.",
     {"analytical": 1.0}),
    ("I enjoy creative work — writing, design, art, or original thinking.",
     {"creative": 1.0}),
    ("I prefer inventing new approaches over following established methods.",
     {"creative": 1.0}),
    ("I feel most energized when collaborating with, teaching, or helping others.",
     {"people_facing": 1.0}),
    ("I'd rather work in a team environment than independently on most tasks.",
     {"people_facing": 1.0}),
    ("I prefer having full ownership of my work and making my own decisions.",
     {"independent": 1.0}),
    ("I work best when I can set my own pace without close supervision.",
     {"independent": 1.0}),
    ("I'm comfortable taking big risks if the potential reward is high enough.",
     {"risk_tolerance": 1.0}),
    ("I'd rather bet on an uncertain high-upside outcome than a safe, predictable one.",
     {"risk_tolerance": 1.0}),
    ("Prestige, status, and being recognized as successful are important to me.",
     {"prestige": 1.0}),
    ("I'm motivated by competition — I want to be among the best in my field.",
     {"prestige": 1.0}),
    ("I prefer environments with clear rules, defined processes, and predictable routines.",
     {"structured": 1.0}),
    ("I like having clear expectations and measurable goals for my work.",
     {"structured": 1.0}),
    ("Making a meaningful difference in people's lives or society is a top priority for me.",
     {"impact": 1.0}),
    ("I'd trade a higher salary for work I know is making the world better.",
     {"impact": 1.0}),
    ("I enjoy going deep into a specialized technical domain and becoming an expert.",
     {"technical_depth": 1.0}),
    ("I prefer mastering hard technical skills over developing general management abilities.",
     {"technical_depth": 1.0}),
    ("I'm excited by the idea of building something from scratch — a company, product, or organization.",
     {"entrepreneurial": 1.0}),
    ("I'd rather create my own opportunity than climb a ladder someone else built.",
     {"entrepreneurial": 1.0}),
]


def answers_to_vector(answers: list[int]) -> np.ndarray:
    """Map 20 answers (each 1-5) to a 10-dim feature vector on the 1-5 scale."""
    if len(answers) != len(QUESTIONS):
        raise ValueError(f"Expected {len(QUESTIONS)} answers, got {len(answers)}")

    feat_idx    = {f: i for i, f in enumerate(FEATURES)}
    totals      = [0.0] * len(FEATURES)
    weight_sums = [0.0] * len(FEATURES)

    for answer, (_, weights) in zip(answers, QUESTIONS):
        for feat, w in weights.items():
            totals[feat_idx[feat]]      += answer * w
            weight_sums[feat_idx[feat]] += w

    # Weighted average per feature. Inputs are 1-5, so the result is already
    # on the same 1-5 scale as the hardcoded career vectors
    raw = [t / s for t, s in zip(totals, weight_sums)]
    return np.array(raw)


# One high-signal clarifying question per feature, shown only when an edge case
# is detected (near-tie, flat profile, or conflicting signals).
TARGETED_QUESTIONS: dict[str, str] = {
    "analytical":      "I prefer to quantify problems and work with data or math rather than intuition.",
    "creative":        "I frequently come up with entirely original ideas rather than refining existing ones.",
    "people_facing":   "I need regular interaction with other people throughout the day to feel engaged.",
    "independent":     "I strongly prefer setting my own direction with minimal oversight.",
    "risk_tolerance":  "I would leave a stable job to pursue something uncertain but exciting.",
    "prestige":        "Having a highly-regarded title or working at a prestigious institution matters a lot to me.",
    "structured":      "I do my best work when I have defined processes and clear expectations to follow.",
    "impact":          "I would accept significantly lower pay to work on something that genuinely helps people.",
    "technical_depth": "I want to become a recognized deep expert in one specialized technical domain.",
    "entrepreneurial": "Building something from the ground up — even if it fails — is more appealing than a safe career.",
}


def apply_targeted_answers(student: np.ndarray, targeted: dict[str, float]) -> np.ndarray:
    """Blend targeted 1-5 scores into an existing 1-5 feature vector at 50/50 weight.

    A pure replacement would ignore the original survey signal entirely; 50/50
    preserves it while letting the clarifying answer nudge the result.
    """
    feat_idx = {f: i for i, f in enumerate(FEATURES)}
    updated  = student.copy()
    for feat, score in targeted.items():
        idx          = feat_idx[feat]
        # score is already on the 1-5 scale, so blend directly.
        updated[idx] = 0.5 * updated[idx] + 0.5 * score
    return updated

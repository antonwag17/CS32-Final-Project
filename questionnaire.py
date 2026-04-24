"""
Student questionnaire mapping to a 10-dim career feature space.

Questions are rated 0-5. A weighted average per feature is linearly
mapped to the 1-5 scale used by career vectors.
"""

import numpy as np

FEATURES = [
    "analytical", "creative", "people_facing", "independent",
    "risk_tolerance", "prestige", "structured", "impact",
    "technical_depth", "entrepreneurial",
]

QUESTIONS: list[tuple[str, dict[str, float]]] = [
    ("I enjoy analyzing data, building models, or solving quantitative problems.",
     {"analytical": 1.0}),
    ("Breaking down complex problems into logical steps is something I genuinely enjoy.",
     {"analytical": 1.0}),
    ("I enjoy creative work — writing, design, art, or original thinking.",
     {"creative": 1.0}),
    ("I prefer inventing new approaches over following established methods.",
     {"creative": 0.6, "entrepreneurial": 0.4}),
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
    """Map N answers (each 0-5) to a 10-dim vector on the 1-5 scale."""
    if len(answers) != len(QUESTIONS):
        raise ValueError(f"Expected {len(QUESTIONS)} answers, got {len(answers)}")
    feat_idx = {f: i for i, f in enumerate(FEATURES)}
    totals = [0.0] * len(FEATURES)
    weight_sums = [0.0] * len(FEATURES)
    for answer, (_, weights) in zip(answers, QUESTIONS):
        for feat, w in weights.items():
            totals[feat_idx[feat]] += answer * w #weighted sum
            weight_sums[feat_idx[feat]] += w
    raw = [t / s for t, s in zip(totals, weight_sums)]
    return 1.0 + (4.0 / 5.0) * np.array(raw)


def prompt_answers() -> list[int]:
    """Ask the user each question on the console; return a list of 0-5 ints."""
    print("\nRate each statement from 0 (strongly disagree) to 5 (strongly agree).\n")
    answers: list[int] = []
    for i, (question, _) in enumerate(QUESTIONS, start=1):
        while True:
            raw = input(f"  {i:2d}. {question}\n      > ").strip()
            try:
                score = int(raw)
            except ValueError:
                print("      Invalid input — enter a whole number.")
                continue
            if 0 <= score <= 5:
                answers.append(score)
                break
            print("      Please enter a number between 0 and 5.")
    return answers


if __name__ == "__main__":
    print("all-0 answers →", answers_to_vector([0] * len(QUESTIONS)).round(2))
    print("all-5 answers →", answers_to_vector([5] * len(QUESTIONS)).round(2))
    print("features      →", FEATURES)

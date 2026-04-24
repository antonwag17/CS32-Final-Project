"""
  1. Ask the student the questionnaire.
  2. Convert answers into a 10-dim feature vector on a 1-10 scale.
  3. Assign the student to the nearest archetype centroid.
  4. Within that archetype, find the single closest career by Euclidean distance.
  5. Display the archetype, best-match career, and the student's strongest traits.
"""

import numpy as np

from questionnaire import FEATURES, prompt_answers, answers_to_vector
from categorize_careers import ARCHETYPE_NAMES, CENTROIDS, CAREERS


def main() -> None:
    print("=" * 60)
    print("                     Career-Match")
    print("=" * 60)

    answers = prompt_answers()
    student = answers_to_vector(answers)

    cluster_id = int(np.argmin(np.linalg.norm(CENTROIDS - student, axis=1)))
    archetype_name = ARCHETYPE_NAMES[cluster_id]

    cluster_careers = [(name, np.array(scores)) for name, cid, scores in CAREERS if cid == cluster_id]
    best_career = min(cluster_careers, key=lambda x: np.linalg.norm(x[1] - student))[0]

    print("\n" + "=" * 60)
    print(f"  Your career archetype:  {archetype_name}")
    print(f"  Best career match:      {best_career}")
    print("=" * 60)

    top3 = sorted(zip(FEATURES, student), key=lambda x: x[1], reverse=True)[:3]
    print("\n  Your strongest traits: " + ", ".join(f.replace("_", " ") for f, _ in top3))


if __name__ == "__main__":
    main()

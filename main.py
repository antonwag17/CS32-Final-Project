# matcher.py

import math

# ── your attribute dimensions ──
ATTRIBUTES = [
    "analytical",
    "creative",
    "social",
    "technical",
    "business",
    "scientific",
]

# ── career paths with handcrafted vectors for now ──
# later these will come from k-means clustering of real job data
CAREER_PATHS = [
    {"title": "Data Science",         "vector": [5, 1, 2, 5, 2, 4]},
    {"title": "Software Engineering", "vector": [4, 2, 2, 5, 3, 1]},
    {"title": "Marketing",            "vector": [2, 5, 5, 1, 4, 1]},
    {"title": "Finance",              "vector": [4, 2, 3, 2, 5, 1]},
    {"title": "UX Design",            "vector": [2, 5, 4, 3, 2, 1]},
    {"title": "Research/Academia",    "vector": [5, 3, 2, 3, 1, 5]},
    {"title": "Consulting",           "vector": [4, 3, 4, 3, 5, 2]},
    {"title": "Healthcare",           "vector": [3, 2, 4, 3, 2, 5]},
]


# ── step 1: collect student profile via questionnaire ──
def build_student_vector():
    print("\nRate yourself from 0 (not at all) to 5 (very much) on each quality.\n")
    vector = []
    for attribute in ATTRIBUTES:
        while True:
            try:
                score = int(input(f"  {attribute}: "))
                if 0 <= score <= 5:
                    vector.append(score)
                    break
                else:
                    print("  Please enter a number between 0 and 5.")
            except ValueError:
                print("  Invalid input — enter a whole number.")
    return vector


# ── step 2: cosine similarity between two vectors ──
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a ** 2 for a in v1))
    mag2 = math.sqrt(sum(b ** 2 for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)


# ── step 3: score student vector against every career path ──
def find_matches(student_vector, career_paths, top_n=3):
    scores = []
    for career in career_paths:
        sim = cosine_similarity(student_vector, career["vector"])
        scores.append((career["title"], round(sim, 3)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


# ── step 4: display results ──
def display_matches(matches):
    print("\nYour top career matches:\n")
    for i, (title, score) in enumerate(matches, start=1):
        bar = "█" * int(score * 20)   # simple visual bar
        print(f"  {i}. {title:25s} {bar} {score}")


# ── main ──
def main():
    print("=== Career Path Matcher ===")
    student_vector = build_student_vector()
    matches = find_matches(student_vector, CAREER_PATHS)
    display_matches(matches)

if __name__ == "__main__":
    main()

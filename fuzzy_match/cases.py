passing_cases = [
    ("comma-shaped", "curved rod"),
    ("eosinophilic", "pink"),
    ("eosinophilic", "red"),
    ("eosinophilic", "pinkish red"),
    ("basophilic", "blue"),
    ("basophilic", "blue-purple"),
    ("basophilic", "bluish purple"),
    ("hypertonic", "tight muscle"),
    ("pleomorphic", "variable shape"),
    ("fluke", "trematode"),
    ("spiral bacilli", "S-shaped bacillus"),
    (
        "numerous multicolored bruises",
        "multiple bruises in various stages of healing",
    ),
]

failing_cases = [
    ("basophilic", "red"),
    ("basophilic", "pink"),
    ("basophilic", "pinkish red"),
    ("hypotonic", "tight muscle"),
]

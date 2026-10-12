from hr_graph_corrective import ask

tests = [
    ("Multi-section, answerable",
     "I'm in my probation period and want to resign. How much notice do I need to give, and would that change if I did it after probation as a Band 3 employee?",
     "1 week during probation; 4 weeks after probation for Bands 1-3"),
    ("Single-section, numeric",
     "How much can I claim for meals on a business trip, and what is the hotel cap in a standard city?",
     "$60 per day for meals, $250 per night for hotels"),
    ("Unanswerable, handbook is silent",
     "If I'm on a Performance Improvement Plan and also in my probation period, which process applies?",
     "Handbook does not say; should refer the employee to HR"),
]

for label, question, expected in tests:
    print("=" * 70)
    print(label)
    print("EXPECTED:", expected)
    print("GOT:", ask(question))
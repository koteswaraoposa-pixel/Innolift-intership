birth_year = int(input("Birth year: "))

if birth_year > 2026:
    print("Error: Invalid year")
else:
    age = 2026 - birth_year
    print(f"Age: {age}")
    print(f"Age in 10 years: {age + 10}")
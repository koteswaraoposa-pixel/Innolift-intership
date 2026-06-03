import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv("student-mat.csv", sep=";")

# 2. Display basic information
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

# 3. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Average G3 grade by study time
avg_studytime = df.groupby("studytime")["G3"].mean()

# 5. Average G3 grade by sex
avg_sex = df.groupby("sex")["G3"].mean()

# 6. Top 5 students by G3
top5 = df.nlargest(5, "G3")

# 7. Additional analysis
print("\nMaximum G3 Grade:", df["G3"].max())
print("Minimum G3 Grade:", df["G3"].min())
print("Mean G3 Grade:", round(df["G3"].mean(), 2))
print("Median G3 Grade:", df["G3"].median())
print("Standard Deviation:", round(df["G3"].std(), 2))

# 8. Pass/Fail Status
df["Result"] = np.where(df["G3"] >= 10, "Pass", "Fail")

# 9. Count Results
print("\nPass/Fail Count:")
print(df["Result"].value_counts())

# 10. Average G3 by School
print("\nAverage G3 by School:")
print(df.groupby("school")["G3"].mean())

# 11. Average G3 by Internet Access
print("\nAverage G3 by Internet Access:")
print(df.groupby("internet")["G3"].mean())

# 12. Students above average
above_avg = df[df["G3"] > df["G3"].mean()]
print("\nStudents Above Average Grade:", len(above_avg))

# 13. Students below average
below_avg = df[df["G3"] < df["G3"].mean()]
print("Students Below Average Grade:", len(below_avg))

# 14. Highest Study Time Group
best_group = avg_studytime.idxmax()
print("\nBest Study Time Group:", best_group)

# 15. Gender with Higher Average Grade
best_gender = avg_sex.idxmax()
print("Better Performing Gender:", best_gender)

# ===== SUMMARY REPORT =====
print("\n========== SUMMARY REPORT ==========")

print("\nAverage G3 Grade by Study Time:")
print(avg_studytime)

print("\nAverage G3 Grade by Sex:")
print(avg_sex)

print("\nTop 5 Students by G3:")
print(top5[["sex", "age", "studytime", "G3"]])

print("\n====================================")
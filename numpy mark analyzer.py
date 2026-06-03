import numpy as np

# NumPy array of 10 student marks
marks = np.array([78, 85, 45, 92, 67, 51, 39, 88, 73, 56])

# Statistics
mean_marks = marks.mean()
highest_marks = marks.max()
lowest_marks = marks.min()
std_dev = marks.std()

# Count passed students (marks >= 50)
passed = np.sum(marks >= 50)

# Summary Report
print("===== STUDENT MARKS SUMMARY REPORT =====")
print("Marks:", marks)
print("Mean Marks:", round(mean_marks, 2))
print("Highest Mark:", highest_marks)
print("Lowest Mark:", lowest_marks)
print("Standard Deviation:", round(std_dev, 2))
print("Students Passed:", passed)
print("Students Failed:", len(marks) - passed)
def get_grade(marks):
    if marks >= 90:
        return "A"
    elif marks >= 75:
        return "B"
    elif marks >= 60:
        return "C"
    elif marks >= 45:
        return "D"
    else:
        return "F"

# Call the function for 5 students
for i in range(1, 6):
    marks = int(input(f"Enter marks for Student {i}: "))
    grade = get_grade(marks)
    print(f"Student {i} Grade: {grade}")
# Function to separate even and odd numbers

def sort_numbers(numbers):
    even = []
    odd = []

    for num in numbers:
        if num % 2 == 0:
            even.append(num)
        else:
            odd.append(num)

    return even, odd

# Test with 10 numbers
nums = [12, 7, 25, 18, 9, 30, 41, 56, 13, 22]

even_numbers, odd_numbers = sort_numbers(nums)

print("Even Numbers:", even_numbers)
print("Odd Numbers:", odd_numbers)
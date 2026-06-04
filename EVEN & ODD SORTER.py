def sort_numbers(numbers):
    even_numbers = []
    odd_numbers = []

    for num in numbers:
        if num % 2 == 0:
            even_numbers.append(num)
        else:
            odd_numbers.append(num)

    return even_numbers, odd_numbers

# Test with 10 numbers
nums = [12, 7, 5, 18, 23, 40, 11, 8, 15, 20]

even, odd = sort_numbers(nums)

print("Original List:", nums)
print("Even Numbers:", even)
print("Odd Numbers:", odd)
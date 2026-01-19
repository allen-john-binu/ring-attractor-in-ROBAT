import matplotlib.pyplot as plt
import numpy as np

filename = "two_target.csv"


data = np.loadtxt(filename, delimiter=",")


array1 = data[:, 101]
array2 = data[:, 102]

# Reorder arrays based on your rules
new_array1 = []
new_array2 = []

for i in range(len(array1)):
    a = array1[i]
    b = array2[i]
    if i <= 175:
        # Case 1: One positive, one negative
        if (a >= 0 and b < 0) or (a < 0 and b >= 0):
            pos = a if a >= 0 else b
            neg = b if a >= 0 else a
            new_array1.append(pos)
            new_array2.append(neg)

        # Case 2: Both negative
        elif a < 0 and b < 0:
            if abs(a) >= abs(b):
                new_array1.append(a)
                new_array2.append(b)
            else:
                new_array1.append(b)
                new_array2.append(a)

        # Case 3: Both positive
        else:  # a >= 0 and b >= 0
            if a >= b:
                new_array1.append(a)
                new_array2.append(b)
            else:
                new_array1.append(b)
                new_array2.append(a)
    else:
        # --- After index 175: array2 gets highest absolute value ---
        if abs(a) >= abs(b):
            new_array1.append(b)
            new_array2.append(a)
        else:
            new_array1.append(a)
            new_array2.append(b)

# Print results
print("Reordered Array 1:", new_array1)
print("Reordered Array 2:", new_array2)

# Plot the result as a point chart
x = range(len(new_array1))

plt.scatter(x, new_array1, label="Array 1", color="blue")
plt.scatter(x, new_array2, label="Array 2", color="green")

plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Reordered Arrays Point Chart")
plt.legend()
plt.grid(True)
plt.show()

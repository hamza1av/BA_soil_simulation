import keyboard
import matplotlib.pyplot as plt

# Initialize a dictionary to count the number of times each key is pressed
key_counts = {}

# Define a callback function to count the key presses
def count_key_press(event):
    if event.name not in key_counts:
        key_counts[event.name] = 0
    key_counts[event.name] += 1

# Register the callback function for all key events
keyboard.on_press(count_key_press)

# Wait for the user to press the 'q' key to quit
keyboard.wait('q')

# Plot the results as a histogram
keys = list(key_counts.keys())
counts = [key_counts[key] for key in keys]
plt.bar(keys, counts)
plt.xlabel('Key')
plt.ylabel('Number of times pressed')
plt.show()

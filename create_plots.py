import matplotlib.pyplot as plt

# lk
y = [12.5, 12.73, 15.2, 18.12, 20.06, 27.79]
x = [3, 9, 17, 25, 33, 37]
plt.plot(x, y, 'g',)

plt.xlabel("Window size")
plt.ylabel("Time in seconds")
plt.legend()
plt.show()
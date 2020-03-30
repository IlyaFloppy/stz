from matplotlib import pyplot as plt
import math

params = list(map(float, open("params.txt", "r").read().split()))
r0 = params[0]
rc = params[1]
k = params[2]
t0 = params[3]
loss = params[4]

def z_of_t(t):
    r = r0 * math.pow(10, k*(t-t0))
    z = 1024 * r / (rc + r)

    return z

z = [27, 31, 43, 58, 69, 86, 102, 111, 122, 137, 18, 87]
t = [71, 64, 52, 41, 33, 23, 17, 12, 2, 0, 87, -5]

z_hat = list([z_of_t(ti) for ti in t])

# sort z, z_hat, t according to t
z = [i for _, i in sorted(zip(t, z))]
z_hat = [i for _, i in sorted(zip(t, z_hat))]
t = sorted(t)

plt.plot(t, z, 'r^', label="observations")
plt.plot(t, z_hat, 'b-', label="model")
plt.plot([], [], label="loss={}".format(loss))
plt.legend(loc="upper right")
plt.xlabel("t")
plt.ylabel("z")
plt.grid()
plt.savefig("plot.png")
import scipy.special as sp
import scipy.constants as cons
import matplotlib.pyplot as plt
import numpy as np


def task_1():
    # Spherical Bessel Functions up to order 3
    f = lambda m: sp.riccati_jn(3, m)[0]
    #  Fresnel Integral
    g = lambda m: sp.fresnel(m)
    x = np.linspace(0, 100, 10000)
    y = [(lambda d: f(d))(d) for d in x]
    y_1 = [(lambda d: g(d))(d) for d in x]
    fig, ax = plt.subplots(figsize=(10, 15))
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
    lineObjects = plt.plot(x, y, label=["$J_0$", "$J_1$", "$J_2$", "$J_3$"])
    plt.xlim(x.min(), x.max() * 0.15)
    lineObjects_2 = plt.plot(x, y_1, label=["Fresnel\quad Sine", "Fresnel\quad Cosine"])
    # plt.legend(lineObjects, ["$J_0$", "$J_1$", "$J_2$", "$J_3$"])
    plt.legend()
    plt.title("Sferyczne Funkcje Bessla $J_n$ i Całka Fresnela $\mathcal{F}$")
    plt.savefig("bessel_jn.pdf", format="pdf")
    plt.show()


def task_2():
    n = 4000
    A = np.zeros((n, n))
    x = np.zeros(n)
    for i in range(n):
        x[i] = i / 2.0
        for j in range(n):
            A[i, j] = i + j + (i + 1) % (j + 1)

    b = np.dot(A, x)
    y = np.linalg.solve(A, b)
    res = x - y
    norm = np.linalg.norm(res)

    print("The norm of resulting vector 'res' is approximately {:.3e}".format(norm))


def task_3A(mean, sigma, N):
    x = np.asarray(
        [(lambda n: np.random.normal(loc=mean, scale=sigma))(n) for n in range(N)]
    )
    mu = np.mean(x)
    sig = np.std(x)

    fig, ax = plt.subplots(figsize=(10, 15))
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
    plt.hist(
        x,
        bins=int(sigma),
        label="Random numbers; $\mu$ = {}, $\sigma$={}, N={}".format(mean, sigma, N),
    )
    plt.vlines(mu, 0, 0.8 * x.max(), label="mean = {:.3e}".format(mu), colors="red")
    plt.hlines(
        0.4 * x.max(),
        mu - sig,
        mu + sig,
        label="std = {:.3e}".format(sig),
        colors="orange",
    )
    plt.legend()
    plt.title("Histogram of gaussian distributed data - Task 3A", fontsize=30)
    plt.xlabel("Random float", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.show()


def task_3B(mean, sigma, N, P):
    xsum = np.zeros(N)
    for i in range(P):
        z = np.array([(lambda n: np.random.uniform(-1, 1))(n) for n in range(N)])
        xsum = np.add(xsum, z, dtype=float)

    xnew = mean + sigma * xsum * np.sqrt(3.0 / P)

    mu = np.mean(xnew)
    sig = np.std(xnew)

    fig, ax = plt.subplots(figsize=(10, 15))
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
    plt.hist(xnew, bins=int(sigma), label="Random numbers; N = {}, P = {}".format(N, P))
    plt.vlines(mu, 0, 0.8 * xnew.max(), label="mean = {:.3e}".format(mu), colors="red")
    plt.hlines(
        0.4 * xnew.max(),
        mu - sig,
        mu + sig,
        label="std = {:.3e}".format(sig),
        colors="orange",
    )
    plt.legend()
    plt.title("Histogram of gaussian distributed data - Task 3B", fontsize=30)
    plt.xlabel("Random float", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.show()


if __name__ == "__main__":
    mean = 15
    sigma = 100
    N = 10000
    P = 10
    task_1()
    task_2()
    task_3A(mean, sigma, N)
    task_3B(mean, sigma, N, P)

from operator import mod
import numpy as np


def SS(N1, N2):
    CellID = 3*N1 + N2

    # PSS
    x = np.zeros(127)
    x[0:7] = [0, 1, 1, 0, 1, 1, 1]
    for i in range(0, 120):
        x[i + 7] = mod(x[i + 4] + x[i], 2)

    PSS = np.zeros(127)
    for n in range(0, 127):
        m = mod(n + 43*N2, 127)
        PSS[n] = 1 - 2*x[m]

    # SSS
    x0 = np.zeros(127)
    x1 = np.zeros(127)
    x0[0:7] = x1[0:7] = [1, 0, 0, 0, 0, 0, 0]
    for i in range(0, 120):
        x0[i + 7] = mod(x0[i + 4] + x0[i], 2)
        x1[i + 7] = mod(x1[i + 1] + x1[i], 2)

    m0 = 15*int(np.floor(N1/112)) + 5*N2
    m1 = mod(N1, 112)
    SSS = np.zeros(127)
    for n in range(0, 127):
        x0_index = mod(n+m0, 127)
        x1_index = mod(n+m1, 127)
        SSS[n] = (1 - 2 * x0[x0_index]) * (1 - 2 * x1[x1_index])

    return [CellID, PSS, SSS]

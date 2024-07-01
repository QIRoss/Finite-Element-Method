import numpy as np

NE = 25
ND = 21
NP = 15

NL = np.array([
    [1, 2, 7], [2, 8, 7], [2, 3, 8], [3, 9, 8], [3, 4, 9],
    [4, 10, 9], [4, 5, 10], [5, 11, 10], [5, 6, 11], [7, 8, 12],
    [8, 13, 12], [8, 9, 13], [9, 14, 13], [9, 10, 14], [10, 15, 14],
    [10, 11, 15], [12, 13, 16], [13, 17, 16], [13, 14, 17], [14, 18, 17],
    [14, 15, 18], [16, 17, 19], [17, 20, 19], [17, 18, 20], [19, 20, 21]
]) - 1

X = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 
              0.2, 0.4, 0.6, 0.0, 0.2, 0.4, 0.0, 0.2, 0.0])

Y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4,
              0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.8, 0.8, 1.0])

NDP = np.array([1, 2, 3, 4, 5, 6, 11, 15, 18, 20, 21, 19, 16, 12, 7]) - 1
VAL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 100.0, 100.0, 100.0, 100.0, 50.0, 
                0.0, 0.0, 0.0, 0.0])

B = np.zeros((ND, 1))
C = np.zeros((ND, ND))

for I in range(NE):
    K = NL[I, :3]
    XL = X[K]
    YL = Y[K]
    P = np.zeros(3)
    Q = np.zeros(3)
    P[0] = YL[1] - YL[2]
    P[1] = YL[2] - YL[0]
    P[2] = YL[0] - YL[1]
    Q[0] = XL[2] - XL[1]
    Q[1] = XL[0] - XL[2]
    Q[2] = XL[1] - XL[0]
    AREA = 0.5 * abs(P[1] * Q[2] - Q[1] * P[2])
    CE = (np.outer(P, P) + np.outer(Q, Q)) / (4.0 * AREA)

    for J in range(3):
        IR = NL[I, J]
        IFLAG1 = 0
        for K in range(NP):
            if IR == NDP[K]:
                C[IR, IR] = 1.0
                B[IR] = VAL[K]
                IFLAG1 = 1
        if IFLAG1 == 0:
            for L in range(3):
                IC = NL[I, L]
                IFLAG2 = 0
                for K in range(NP):
                    if IC == NDP[K]:
                        B[IR] -= CE[J, L] * VAL[K]
                        IFLAG2 = 1
                if IFLAG2 == 0:
                    C[IR, IC] += CE[J, L]

V = np.linalg.solve(C, B)

print("ND, NE, NP:", ND, NE, NP)
print("Nós, X, Y, V:")
for i in range(ND):
    print(i + 1, X[i], Y[i], V[i, 0])
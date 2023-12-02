import numpy as np
from typing import List


def generate_G(k: int) -> np.array:
    G = np.zeros((2 ** k, k + 1))

    number_of_1_sequential = 2 ** k
    G[:, 0] = 1
    for i in range(1, k + 1):
        number_of_1_sequential //= 2
        is_1 = True
        for j in range(0, 2 ** k, number_of_1_sequential):
            if is_1:
                G[j:j + number_of_1_sequential, i] = 1
                is_1 = False
            else:
                G[j:j + number_of_1_sequential, i] = 0
                is_1 = True
    return G


def generate_H_from_G(G: np.array, n: int) -> np.array:
    k = int(np.log2(n)) + 1
    G = G[:n, ...]
    alpha_vectors = generate_G(k + 1)[:, 1:]

    H = np.zeros((n, 2 ** (k + 1)))
    for column, alpha_vector in enumerate(alpha_vectors):
        H[:, column] = np.sum(alpha_vector[None, :] * G, axis=1) % 2

    return H


def generate_H(n: int) -> np.array:
    k = int(np.log2(n)) + 1
    G = generate_G(k)
    return generate_H_from_G(G, n)


def count_knf_true_dis(knf: List[List[int]], variables: List[int]) -> int:
    count = 0
    for dis in knf:
        for variable in dis:
            actual_value = variables[abs(variable) - 1]
            if variable < 0:
                actual_value = (actual_value + 1) % 2
            if actual_value == 1:
                count += 1
                break
    return count


if __name__ == "__main__":
    line1 = input().strip().split(" ")
    n, m = int(line1[0]), int(line1[1])

    knf = []
    for i in range(m):
        line = input().strip().split(" ")
        knf.append([int(variable) for variable in line])

    H = generate_H(n).T
    for row in H[::-1]:
        if count_knf_true_dis(knf, row.tolist()) >= (7/8) * m:
            print(''.join([str(int(value)) for value in row.tolist()]))
            break

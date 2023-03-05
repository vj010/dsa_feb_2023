import sys
import numpy as np
from collections import Counter
sys.stdin = open('plus_minus/input.txt', 'r')
sys.stdout = open('plus_minus/output.txt', 'w')


def matrix_exponentiaton(arr: np.ndarray, y) -> np.ndarray:
    M = np.array([[0, 1], [1, 1]])
    ans = np.array([[1, 0], [0, 1]])
    while y > 0:
        if y & 1:
            ans = np.matmul(ans, M)
        M = np.matmul(M, M)
        y >>= 1
    return np.matmul(arr, ans)


def unique_solution_exists(a1: int, b1: int, a2: int, b2: int) -> bool:
    x1 = a1*b2
    x2 = a2*b1
    if x1 != x2:
        # print("unique solution exists: x1:{}, x2:{}".format(x1, x2))
        return True
    return False


def get_y(a1: int, b1: int, c1: int, a2: int, b2: int, c2: int) -> float:
    return (c2*a1 - a2*c1)/((a1*b2)-(a2*b1))


def get_x(a1: int, b1: int, c1: int, y: float) -> float:
    return (c1-b1*y)/a1


def solve_linear_equation_in_two_variable(a1: int, b1: int, c1: int, a2: int, b2: int, c2: int) -> np.ndarray:

    if not unique_solution_exists(a1, b1, a2, b2):
        return np.array([-1])

    y = get_y(a1, b1, c1, a2, b2, c2)
    x = get_x(a1, b1, c1, y)

    if int(x) != x or int(y) != y:
        return np.array([-1])

    # print("x:{}, y:{}".format(x, y))
    return np.array([x, y])


def verify_array(a: int, b: int, given_array: np.ndarray) -> bool:
    seed_a = max(a, b)
    seed_b = min(a, b)
    new_arr = np.zeros(np.shape(given_array))
    new_arr[0] = seed_a
    new_arr[1] = seed_b
    new_arr[2] = seed_a+seed_b
    new_arr[3] = seed_a - seed_b
    remaining_elements_count = int(np.size(given_array))
    for i in range(4, remaining_elements_count, 2):
        new_arr[i] = new_arr[i-2] + new_arr[i-4]
        new_arr[i+1] = new_arr[i-2] - new_arr[i-4]

    # print("new arr:{}".format(new_arr))

    new_arr_freq = Counter(new_arr)
    given_array_freq = Counter(given_array)

    # print(new_arr_freq)
    # print(given_array_freq)
    for key, val in new_arr_freq.items():
        if key not in given_array_freq or given_array_freq[key] != val:
            return False

    return True


def solve(arr_size: int, first_largest_element: int, second_largest_element: int, arr: np.ndarray) -> None:
    [a1, a2] = matrix_exponentiaton(np.array([1, 1]), int(arr_size/2)-2)
    [b1, b2] = matrix_exponentiaton(np.array([0, 1]), int(arr_size/2)-2)
    # print("a1:{}, b1:{}, a2:{},b2:{}".format(a1, b1, a2, b2))
    res = solve_linear_equation_in_two_variable(
        a1, b1, second_largest_element, a2, b2, first_largest_element)
    # print(res)
    res = res.astype(int)
    if np.size(res) < 2 or res[0] < 0 or res[1] < 0:
        # print('here')
        print('NO')
    else:
        if verify_array(res[0], res[1], arr):
            print('YES')
            print(res[0], res[1])
        else:
            print('NO')


t = int(sys.stdin.readline().split()[0])

while t > 0:
    n = int(sys.stdin.readline().split()[0])
    arr = np.array(list(map(int, sys.stdin.readline().split())))
    np.ndarray.sort(arr)
    first_largest = arr[-1]
    second_largest = arr[-2]
    # print(first_largest)
    # print(second_largest)
    solve(n, first_largest_element=first_largest,
          second_largest_element=second_largest, arr=arr)

    t -= 1

initial_fib_1 = np.array([1, 1], dtype=int)
initial_fib_2 = [0, 1]
# print(matrix_exponentiaton(initial_fib_1, 2))
# test_arr = np.array([1, 2])
# np.append(test_arr, [4], axis=0)
# print("test_arr:{}".format(test_arr))

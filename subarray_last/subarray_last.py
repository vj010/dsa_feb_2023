import sys
import numpy as np
from collections import Counter
sys.stdin = open('subarray_last/input.txt', 'r')
sys.stdout = open('subarray_last/output.txt', 'w')


def calculate_val(arr: np.ndarray) -> int:
    ans = np.sum(np.array([(i+1)*arr[i] for i in range(np.size(arr))]))
    return ans


t = int(sys.stdin.readline().split()[0])
while t > 0:
    n = int(sys.stdin.readline().split()[0])
    arr = np.array(list(map(int, sys.stdin.readline().split())))
    print(calculate_val(arr))
    t -= 1

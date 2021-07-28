import datetime
import itertools
import random
import collections
import string
import math
import copy
import os
import sys
from io import BytesIO, IOBase

BUFSIZE = 8192

class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")


# sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)


# def input(): return sys.stdin.readline().rstrip("\r\n")

from fractions import Fraction

def read_int():
    s = input().split()
    if len(s) == 1:
        return int(s[0])
    else:
        return map(int, s)

def read_list():
    return [int(i) for i in input().split()]


def exgcd(a, b):
    if not b:
        return 1, 0
    y, x = exgcd(b, a % b)
    y -= a//b * x
    return x, y


def getinv(a, m):
    x, y = exgcd(a, m)
    return -1 if x == 1 else x % m


def lagrange(points: list):
    x = len(points)
    cof = [Fraction(0) for i in range(x)]
    num = [Fraction(0) for i in range(x+1)]
    tmp = [Fraction(0) for i in range(x+1)]
    num[0] = Fraction(1)
    for i in range(1, x+1):
        tmp[0] = 0
        for j in range(1, i+1):
            tmp[j] = num[j-1]
        for j in range(i+1):
            tmp[j] += num[j] * -points[i-1][0]
        num, tmp = tmp, num
    for i in range(1, x+1):
        den = Fraction(1)
        lst = Fraction(0)
        for j in range(1, x+1):
            if i!=j:
                den = den * (points[i-1][0] - points[j-1][0])
        den = points[i-1][1] / den
        for j in range(x):
            tmp[j] = (num[j]-lst) * Fraction(1, -points[i-1][0])
            cof[j] += den*tmp[j]
            lst = tmp[j]
    return cof
# mo = 998244353
# n, k = read_int()
# pp = []
# for i in range(n):
#     x, y = read_int()
#     pp.append((x,y))
# cof = lagrange(pp)
# res = 0
# pk = 1
# for i in cof:
#     res += pk * i.numerator%mo * getinv(i.denominator, mo) % mo
#     res%=mo
#     pk*=k
#     pk%=mo
# print(res)

def powersum(x):
    points = [(1, 1)]
    for i in range(2, x+3):
        points.append((i, i**x+points[-1][1]))
    # print(points)
    return lagrange(points)

import math

def lcm(x,y):
    return x*y//math.gcd(x,y)

MP = {
    0:"1 1 0",
    1:"2 1 1 0",
    2:"6 2 3 1 0",
    3:"4 1 2 1 0 0",
    4:"30 6 15 10 0 -1 0",
    5:"12 2 6 5 0 -1 0 0",
    6:"42 6 21 21 0 -7 0 1 0",
    7:"24 3 12 14 0 -7 0 2 0 0",
    8:"90 10 45 60 0 -42 0 20 0 -3 0",
    9:"20 2 10 15 0 -14 0 10 0 -3 0 0",
    10:"66 6 33 55 0 -66 0 66 0 -33 0 5 0",
    11:"24 2 12 22 0 -33 0 44 0 -33 0 10 0 0",
    12:"2730 210 1365 2730 0 -5005 0 8580 0 -9009 0 4550 0 -691 0",
    13:"420 30 210 455 0 -1001 0 2145 0 -3003 0 2275 0 -691 0 0",
    14:"90 6 45 105 0 -273 0 715 0 -1287 0 1365 0 -691 0 105 0",
    15:"48 3 24 60 0 -182 0 572 0 -1287 0 1820 0 -1382 0 420 0 0",
    16:"510 30 255 680 0 -2380 0 8840 0 -24310 0 44200 0 -46988 0 23800 0 -3617 0",
    17:"180 10 90 255 0 -1020 0 4420 0 -14586 0 33150 0 -46988 0 35700 0 -10851 0 0",
    18:"3990 210 1995 5985 0 -27132 0 135660 0 -529074 0 1469650 0 -2678316 0 2848860 0 -1443183 0 219335 0",
    19:"840 42 420 1330 0 -6783 0 38760 0 -176358 0 587860 0 -1339158 0 1899240 0 -1443183 0 438670 0 0",
    20:"6930 330 3465 11550 0 -65835 0 426360 0 -2238390 0 8817900 0 -24551230 0 44767800 0 -47625039 0 24126850 0 -3666831 0"
}

def solve():
    input()
    n = read_int()
    print(MP[n])
    # print()
    # cof = powersum(n)[::-1]
    # M = 1
    # for i in cof:
    #     M = lcm(M, i.denominator)
    # for p, i in enumerate(cof):
    #     cof[p] = (i*M).numerator
    # print(M, *cof)


T = read_int()
for TI in range(T):
    solve()
    if TI!=T-1:
        print()

# print(powersum(1))
# print(powersum(2))
# print(powersum(3))
# print(powersum(4))
# print(powersum(5))
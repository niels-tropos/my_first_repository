from random import random
import math
from re import X
import matplotlib.pyplot as plt

def fun(a):
    steps = []
    steps.append(a)

    if (a % 2) == 0:
        new = a / 2
    else:
        new = 3*a + 1

    steps.append(new)

    while new != 1:

        if (new % 2) == 0:
            new = new / 2
        else:
            new = 3*new + 1

        steps.append(new)

    return steps


def number_generator(n, factor):
    numbers = []
    z = int(math.log(factor,10))

    for i in range(0, n):
        numbers.append(round(random(),z)*factor)

    return numbers


a = number_generator(10,10)

print(a)

x = 9
y = 45
r = y**X
print(r)
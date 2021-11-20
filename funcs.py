import math
from functools import reduce


def _min(data):
    return reduce(lambda x, y: y if y < x else x, data)


def _max(data):
    return reduce(lambda x, y: y if y > x else x, data)


def _count(data):
    count = 0
    for i in data:
        if math.isnan(i):
            continue
        count += 1
    return count


def _mean(data):
    return reduce(lambda x, y: x + y if not math.isnan(y) else x, data) / _count(data)


def _std(data):
    m = _mean(data)
    total = 0
    for x in data:
        if math.isnan(x):
            continue
        total += (x - m) ** 2
    try:
        return math.sqrt(total / _count(data))
    except ZeroDivisionError:
        return math.nan


def _percentile(data, p):
    data.sort()
    k = (_count(data) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return data[int(k)]

    d0 = data[int(f)] * (c - k)
    d1 = data[int(c)] * (k - f)
    return d0 + d1

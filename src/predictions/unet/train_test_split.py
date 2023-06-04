def test():
    counter = 0
    while counter < 5:
        yield (1, 2), (3, 4)
        counter += 1


for idx, ((a, b), (c, d)) in enumerate(test()):
    print(a, b, c, d)

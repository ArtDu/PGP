from random import *

if __name__ == '__main__':
    # low test
    n = 2**6
    name = 'low'
    f = open('tests/' + name + '.t', 'w')
    f.write(f"{n}\n")
    for _ in range(n):
        a = randint(1, 2**30)
        f.write(f"{a} ")
    f.write("\n")
    for _ in range(n):
        a = randint(1, 2**30)
        f.write(f"{a} ")
    f.write("\n")
    f.close()

    # mid test
    n = 2**16
    name = 'mid'
    f = open('tests/' + name + '.t', 'w')
    f.write(f"{n}\n")
    for _ in range(n):
        a = randint(1, 2**30)
        f.write(f"{a} ")
    f.write("\n")
    for _ in range(n):
        a = randint(1, 2**30)
        f.write(f"{a} ")
    f.write("\n")
    f.close()

    # high test
    n = 2**24
    name = 'high'
    f = open('tests/' + name + '.t', 'w')
    f.write(f"{n}\n")
    for _ in range(n):
        a = randint(1, 2**30)
        f.write(f"{a} ")
    f.write("\n")
    for _ in range(n):
        a = randint(1, 2**30)
        f.write(f"{a} ")
    f.write("\n")
    f.close()

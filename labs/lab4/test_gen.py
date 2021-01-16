from random import uniform

if __name__ == '__main__':
    n = 2000
    m = 2000
    name = str(n) + "_" + str(m)
    file = open(name + '.t', 'w')
    file.write("{} {}\n".format(n, m))
    for _ in range(n):
        for _ in range(m):
            file.write("{} ".format(uniform(-0.00005, 0.00005)))
        file.write("\n")
    file.close()
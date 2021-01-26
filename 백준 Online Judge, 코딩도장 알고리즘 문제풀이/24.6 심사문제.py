x = input().split(';')
b = list(map(int,x))

b.sort(reverse=True)

for i in b:
    print('{0:>9}'.format(i, ','))

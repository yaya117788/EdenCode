a, b = input().split()
line = []
line1 = []

for i in reversed(range(len(a))):
    line.append(a[i])

for i in reversed(range(len(b))):
    line1.append(b[i])

c = int(''.join(line))
d = int(''.join(line1))

if c > d:
    print(c)
else :
    print(d)






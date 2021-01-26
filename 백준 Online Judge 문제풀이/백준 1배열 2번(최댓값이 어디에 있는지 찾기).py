a = []
for i in range(9):
    b = int(input())
    a.append(b)
maxvalue = a[0]
for i in a:
    if i > maxvalue:
        maxvalue =i

print(maxvalue)
print(a.index(maxvalue)+1)

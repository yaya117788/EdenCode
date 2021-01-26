a = int(input())
d = a

for i in range(a):
    b = input()
    for j in range(len(b)):
        if b[j] in b[j+1:]:
            c = b.count(b[j])
            if b[j] != b[b.find(b[j])+(c-1)]:
                d -= 1
                break
            else:
                if (b.rfind(b[j])-b.index(b[j])) >= b.count(b[j]) :
                    d -= 1
                    break
print(d)

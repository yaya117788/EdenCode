x, b = map(int,input().split())
if 1 <= x <=20 and 10 <= b <=30:
    a = list(2**i for i in range(x,b+1))
    del a[1],a[-2]
    print(a)


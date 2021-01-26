a, b = map(int,input().split())

while 0 < a and b <10:
    print(a+b)
    a, b = map(int,input().split())
    if a == 0 and b == 0:
        break


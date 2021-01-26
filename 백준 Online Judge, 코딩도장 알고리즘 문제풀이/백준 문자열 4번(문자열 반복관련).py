a = int(input())
b = 0
for i in range(a):
    x, y = input().split()
    x = int(x)
    y = [y[i] for i in range(len(y))]
    b= [''.join(y[j]*x) for j in range(len(y))]

    print(''.join(b))

import sys
a = list(map(int, sys.stdin.readline().split()))
c = a.copy()
c.sort()
b = c.copy()
b.sort(reverse=True)
if a == b :
    print('descending')
elif a == c :
    print('ascending')
else :
    print('mixed')





import sys
n = int(input())

for i in range(1, n+1):
    a , b = map(int, sys.stdin.readline().split())
    if 0 <= a and b <= 10:
        print('Case ','#',i,': ',a,' + ',b,' = ',a+b, sep='')

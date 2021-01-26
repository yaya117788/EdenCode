x = int(input())
avg = 0
b = 0
for i in range(x):
    count = 0
    c = 0
    a = list(map(int, input().split()))
    b = a[0]
    for j in range(b):
        avg = (sum(a)-a[0]) / (len(a)-1)
        if a[j+1] > avg:
            count +=1
    c = (count*100)/(len(a)-1)       
    print('%.3f' % c,'%',sep='')

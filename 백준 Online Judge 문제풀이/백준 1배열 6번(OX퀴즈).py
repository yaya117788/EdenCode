x = int(input())


for i in range(x):
    ind = 0
    count = 0
    a = list(map(str, input()))
    
    for k in range(len(a)):
        if a[k] == 'O':
            count +=1
            ind = ind + count
        else :
            count = 0
    print(ind)

a = int(input())
x = list(map(int, input().split()))
largest = x[0]
for i in x :
    if i > largest:
        largest = i
b= []
for i in x:
    b.append(i/largest*100)
count = 0
for i in range(len(b)):
    count += b[i]
    
print(count/len(b))

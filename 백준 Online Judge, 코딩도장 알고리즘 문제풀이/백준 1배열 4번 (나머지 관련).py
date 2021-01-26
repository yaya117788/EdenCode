list1 = []
for i in range(10):
    a= int(input())
    j = a % 42
    list1.append(j)
    
print(len(set(list1)))

# a =  [-4,-1,0,2,1,-2,-1,4,0,0,3,-3]
a = [-1,0,1,2,-1,-4]

result = []
a.sort()
print(a)
for i in range(len(a)//2):
    left = i+1
    right = len(a) -1
    if i >0 and (a[i] == a[i-1]) :
        continue
    
    while left < right :
        sum_ = a[i] + a[left] + a[right]
        if (sum_ < 0 ):
            left += 1 
        elif (sum_ > 0):
            right -= 1
        else :
            result.append([a[i],a[left],a[right]])
#             while (a[i] +a[left]) <0 :
#                 left += 1
#             while (a[i]+ a[right]) > 0:
#                 right -= 1
            left += 1 
            right -= 1 
print(result)

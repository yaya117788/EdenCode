a = [3,1,6,2,1,4,5]
target = 9

sum_ = 0

for i in range(len(a)):
    sum_ = target - a[i]
    if (sum_) in a[i+1:] :
        # a[i +1 : ] 이런 디테일 주의 하라 
        print(i)
        print(a.index(sum_))
        break

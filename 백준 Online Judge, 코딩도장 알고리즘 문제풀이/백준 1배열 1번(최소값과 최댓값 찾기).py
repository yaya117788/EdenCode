import sys
n = int(input())
a = list(map(int, sys.stdin.readline().split()))
smallest = a[0]

for i in a:
    if i < smallest :
        smallest = i
        
print(smallest)
largest = a[0]
for i in a:
    if i > largest :
        largest = i

print(largest)



# or
print(max(a))
print(min(a))

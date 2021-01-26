a, b  = map(int, input().split())

a = {i for i in range(1,a+1) if a % i == 0}
b = {i for i in range(1,b+1) if b % i == 0}


divisor = x & b
print(divisor)
result = 0
if type(divisor) == set:
    result = sum(divisor)

print(result)


x = int(input())

if 1 <= x <= 4000 :
    if x % 4 == 0 :
        if x % 100 != 0 or x % 400 == 0 :
            x = 1
        else :
            x = 0
    else :
        x = 0

print(x)




h, m = map(int, input().split())

if 0 <= h <= 23 and 0 <= m <= 59 :
    if 45 <= m :
        m -= 45
    elif m <= 45 and 1 <= h :
        h -= 1 
        m = 15 + m
    elif m <= 45 and h == 0 :
        h = 23
        m = 15 + m
print(h,m)

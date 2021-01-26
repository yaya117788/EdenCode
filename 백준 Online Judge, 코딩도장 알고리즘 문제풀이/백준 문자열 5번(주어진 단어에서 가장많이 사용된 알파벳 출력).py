s = str(input())
s = list(s.upper())
largest = s[0]
for i in range(len(s)):
    if s.count(s[i]) > s.count(largest):
        largest = s[i]
for i in range(len(s)):
    if s.count(s[i]) == s.count(largest):
        if s.index(s[i]) == s.index(largest):
            continue
        else:
            largest = '?'
    else:
        largest = largest
print(largest)

    

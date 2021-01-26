a = str(input())
line = []
b ='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
c = 0

if 2 <= len(a) and len(a) <= 15:
    for i in range(len(a)):
        line.append(a[i])

for i in range(len(line)):
    if line[i] in 'ABC':
        c = c + 3
    elif line[i] in 'DEF':
        c = c + 4
    elif line[i] in 'GHI':
        c = c + 5
    elif line[i] in 'JKL':
        c = c + 6
    elif line[i] in 'MNO':
        c = c + 7
    elif line[i] in 'PQRS':
        c = c + 8
    elif line[i] in 'TUV':
        c = c + 9
    elif line[i] in 'WXYZ':
        c = c + 10


print(c)
    

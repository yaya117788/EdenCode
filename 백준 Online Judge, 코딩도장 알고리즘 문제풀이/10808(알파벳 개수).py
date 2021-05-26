alpha = 'abcdefghijklmnopqrstuvwxyz'
ar = list('0'*len(alpha))
word = input()
for i in range(len(word)):
    a = 0
    a = alpha.index(word[i])
    ar[a] = int(ar[a])+ 1
    ar[a] = str(ar[a])
print(' '.join(ar))

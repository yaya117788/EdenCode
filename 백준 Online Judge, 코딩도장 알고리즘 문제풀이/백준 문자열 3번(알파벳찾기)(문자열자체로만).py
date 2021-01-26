alphabet ='abcdefghijklmnopqrstuvwxyz'
x = str(input())
alphabet = [alphabet[i] for i in range(len(alphabet))]
bl = alphabet.copy()
line = []
for i in range(len(alphabet)):
    line.append(-1)
for i in alphabet:
    for j in range(len(x)):
        if i == x[j]:
            if x[j] in alphabet:
                alphabet[alphabet.index(i)] = str(x.index(x[j]))


for i in range(len(line)):
    if alphabet[i] == bl[i]:
        alphabet[i] = str(line[i])
print(' '.join(alphabet))
            


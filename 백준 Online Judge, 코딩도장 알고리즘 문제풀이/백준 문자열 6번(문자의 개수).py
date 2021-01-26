n = input().split()

n = ' '.join(n)
import string

n = n.strip(string.punctuation)
n = n.split()

print(len(n))

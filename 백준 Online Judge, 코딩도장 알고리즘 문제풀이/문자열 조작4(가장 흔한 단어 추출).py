import string
import collections
words = input()
banword = input().split()

words = words.lower().split()
for i in words:
    index = words.index(i)
    words[index] = i.strip(string.punctuation)
    if (i) in banword:
        words.remove(i)

s = collections.Counter(words)
s.most_common(1)[0][0]

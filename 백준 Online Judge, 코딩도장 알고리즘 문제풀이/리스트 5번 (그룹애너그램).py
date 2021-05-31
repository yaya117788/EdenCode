import collections
x = input().split()

dict_ = collections.defaultdict(list)

for word in x:
    dict_[''.join(sorted(word))].append(word)
    
print(dict_)

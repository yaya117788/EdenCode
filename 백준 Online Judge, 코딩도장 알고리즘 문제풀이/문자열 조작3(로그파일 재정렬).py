list1,list2 = [],[]
for i in logs:
    if i.split(' ')[1].isdigit():
        list1.append(i)
    else:
        list2.append(i)
list2.sort(key=lambda x : (x.split()[1:], x.split()[0]))
# or 
'''
def func(x):
    return x.split()[1:], x.split()[0]
x.sort(key = func)도 가능하다 
'''
list2 + list1

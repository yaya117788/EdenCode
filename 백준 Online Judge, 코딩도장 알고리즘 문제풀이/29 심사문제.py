with open('words.txt','r')as file:
    line = file.readlines()
    lines = []
    for i in line:
        lines.append(i.strip('\n'))
    tmp = lines.copy()
    for i in lines:
        for j in range(len(i)//2):
            if i[j] != i[-1-j]:
                tmp.remove(i)
                break            
    for i in tmp:
        print(i)

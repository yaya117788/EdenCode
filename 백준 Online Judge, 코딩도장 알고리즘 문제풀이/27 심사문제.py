with open('words.txt','r')as file:
    line = None
    while line != '':
        line = file.readline()
        lines = line.split()
        a = []
        c = []
        for i in range(len(lines)):
            if 'c' in lines[i]:
                a.append(lines[i])
        for i in a:
            c.append(i.strip(',.'))
        print('\n'.join(c))
            

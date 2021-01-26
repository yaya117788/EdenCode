a = 0
line1 =[]
while True:
    a = str(input())
    if a == '':
        break
    line1.append(a)
print(line1)
with open('words.txt','w') as file:
    file.writelines(line1)

with open('words.txt','r') as file:
    line = None
    while line != '':
        line = file.readline()
        for i in range(len(line) //2):
           if line[i] == line[-1-i]:
              print(line)
        

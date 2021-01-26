a, b = map(int,input().split())
line = []
for i in range(a):
    line.append(list(input()))
for i in range(len(line)):
    for j in range(len(line[i])):
        k = 0
        if line[i][j] == '.':
            try:
               if i < len(line):
                   if line[i+1][j] == '*':
                       k += 1
            except :
                pass
            try:
               if j < len(line[i]):
                   if line[i][j+1] == '*':
                       k += 1
            except :
                pass
            try:
               if i > 0 :
                   if line[i-1][j] == '*':
                       k += 1
            except:
                pass
            try:
               if j > 0 :
                   if line[i][j-1] == '*':
                        k += 1
            except:
                pass
            try:
                if i < len(line) and j < len(line[i]):
                    if line[i+1][j+1] == '*':
                        k += 1
            except:
                pass
            try:
                if i > 0 and j < len(line[i]):
                    if line[i-1][j+1] == '*':
                        k += 1
            except:
                pass
            try:
                if i >0 and j >0:
                    if line[i-1][j-1] == '*':
                        k += 1
            except:
                pass
            try:
                if i < len(line) and j >0:
                    if line[i+1][j-1] == '*':
                        k +=1
            except:
                pass
            line[i][j] = k
            #먼저 한 점에서 주변 지뢰의 개수를 찾아서 그 .을 숫자 값으로 바꿔서 출력되게 해야함
            #pprint로 결국 출력해야 답이나옴
for x in range(len(line)):
    for y in range(len(line[x])):
        print(line[x][y], end='')
    print() # 이것을 왜 쓰는것일까

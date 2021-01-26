t= input()
case2 = 0
case3 = 0

for i in range(len(t)):
    if t[i] == 'j' and i != 0:
        if t[i-1] == 'n' or t[i-1] == 'l':
            case2 +=1
    elif t[i] == '-' and i != 0:
        if t[i-1] == 'c' or t[i-1] == 'd':
            case2 +=1
    elif t[i] == '=' and i != 0:
        if t[i-1] == 'c' or t[i-1] == 's':
            case2 += 1
        else:
            if t[i-1] == 'z' and t[i-2] != 'd':
                case2 +=1
            else:
                case3 +=1

cnt = len(t) - case2 -(case3*2)
print(cnt)

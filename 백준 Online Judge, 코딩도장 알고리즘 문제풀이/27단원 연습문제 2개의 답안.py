with open('words.txt', 'r') as file:
    count = 0
    line = None
    while line != '':
        line = file.readline()
        if len(line.strip('\n')) <= 10:
            count +=1

    print(count)


    ### 또다른 답
    words = file.readlines()
    for word in words:
        if len(word.strip('\n')) <= 10:
            count += 1

korean, english, mathematics, science = 100,85, 81, 91

def get_max_score(*arg):
    a = list(arg)
    highest = a[0]
    for i in a:
        if i >= highest :
            i == highest
    return highest

max_score = get_max_score(korean,english,mathematics,science)
print('높은 점수:',max_score)

max_score = get_max_score(english,science)
print('높은 점수:',max_score)

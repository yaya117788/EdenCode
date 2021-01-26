x = str(input())
#1번 replace로 their them등을 다른걸로 교체후 the만 count로 돌려버리기
#공백에 0을 채워버리고 0the0만을 검색
#split으로 리스트로 분리후 count

table = str.maketrans(',','0')
mark = str.maketrans('.','0')
x = x.translate(table)
x = x.translate(mark)
x = x.replace(' ','0')

print(x.count('the0'))


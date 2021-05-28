x = input()
list_ = []
for i in range(len(x)):
    if x[i].isalnum():
        list_.append(x[i].lower())
# isalnum() 함수는 영소문자, 숫자만을 구분해준다.

a = ''.join(list_)
print(a == a[::-1])

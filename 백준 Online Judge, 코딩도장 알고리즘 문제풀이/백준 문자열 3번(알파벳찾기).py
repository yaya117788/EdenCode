alphabet ='a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'
s = alphabet.split(',')#그냥 'abcd로 했을때 split으로 구분방법이있는지
x = str(input())
a = []
for i in x:  # 문자열을 넣으면 하나씩 들어감
    a.append(i)

for i in range(len(a)):
    if a[i] in s:
        s.index(a[i]).replace(i)
    else:
        s = '-1'
print(s)













#리스트방식 말고도 풀수있는방법이 있는지

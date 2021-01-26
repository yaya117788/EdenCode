def countdown(n):
    number = n+1
    def countnb():
        nonlocal number
        number -= 1
        return number
    return countnb
  



n = int(input())

c = countdown(n)
for i in range(n):
    print(c(), end =' ')

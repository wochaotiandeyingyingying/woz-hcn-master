ss = input().split()
n = int(ss[0])
k = int(ss[1])
str1 = ss[2]
str2 = ss[3]
if str1  == str2:
    print([k,k])
    exit()
#判断是否有相同的
str1 = str1.strip('\"')
str2 = str2.strip('\"')
flag = []
for i in range(n):
    if str1[i] == str2[i]:
        flag.append(i)
if not flag:
    print([0,n-k])
    exit()
#求max
if len(flag) > k:
    max_n = n-(len(flag) - k)
else:
    max_n = n-( k-len(flag))
#求min
if n - len(flag) > k:
    min_n = 0
else:
    min_n =  k-(n - len(flag))
print([min_n,max_n])

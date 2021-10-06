def answer(input=10):
    x = input
    num = 2
    num2besquareddict = {2:2}
    squarednumbers = [[1, 2, 1]]

    while True:
        while True:
            num += 1
            check = [i[2] for i in squarednumbers]
            if num not in check:
                ceiling = num ** 2
                num2besquareddict[num] = 2
                break

        ceilinglist = [j ** num2besquareddict[j] for j in num2besquareddict]
        ceilinglist.append(ceiling)
        ceiling = min(ceilinglist)
            
        tempsqnums = []
        for i in num2besquareddict:
            exponentednum = 0
            while True:
                exponentednum = i ** num2besquareddict[i]
                if exponentednum > ceiling:
                    break
                tempsqnums.append([i, num2besquareddict[i], exponentednum])
                num2besquareddict[i] = num2besquareddict[i] + 1
        
        tempsqnums.sort(key=lambda x: x[2])
        squarednumbers = squarednumbers + tempsqnums
        
        if len(squarednumbers) > x:
            break

    print(squarednumbers)
    print(squarednumbers[x-1][2])

    return squarednumbers[x-1][2]


answer(20)
import datetime
import os, sys, random
import string
charset = string.ascii_uppercase

def randstr(l):
    return ''.join(random.choices(charset, k=l))

def randomtree(siz: int) -> list:
    ret = []
    rem = [i for i in range(1,siz+1)]
    random.shuffle(rem)
    while len(rem)>1:
        rd = rem.pop()
        ret.append((rd, random.choice(rem)))
    return ret

program1 = 'python dd.py'
program2 = 't3.exe'

for i in range(1000000):

    # tc = random.randint(1,4)
    n = random.randint(1, 1000)
    x = random.randint(1, 500)
    y = random.randint(1, 500)


    # lee = random.randint(1, 2000)

    with open('test.data','w') as f:
        # f.write("1\n")
        f.write(f"{n} {x} {y}\n")
        for ti in range(n):
            op = random.randint(1, max(x, y))
            if ti!=0:
                f.write(' ')
            f.write(str(op))
        f.write('\n')
    # print('cpping')
    # os.system('duip < test.data > tcpp.out')
    t1 = datetime.datetime.now()
    tpy = os.popen(f'{program1} < test.data').read().strip()
    t1 = datetime.datetime.now() - t1
    # print('pying')
    # os.system('dd.py < test.data > tpy.out')
    t2 = datetime.datetime.now()
    tcpp = os.popen(f'{program2} < test.data').read().strip()
    t2 = datetime.datetime.now() - t2
    # with open('tpy.out','r') as f:
    #     tpy = f.read()
    # with open('tcpp.out', 'r') as f:
    #     tcpp = f.read()
    print(f"{t1.total_seconds()} / {t2.total_seconds()}")
    if tpy!=tcpp:
        # print(f'ERROR OCCURED at s1={s1} with s2={s2}')
        print(f"{program1} ====>")
        print(tpy)
        print('DIFFERS =====')
        print(tcpp)
        print(f"{program2} <====")
        break
    if i%1000 == 0:
        print(f'testing {i+1}th data...')
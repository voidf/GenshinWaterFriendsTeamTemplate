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

exe1 = 'norm.exe'

exe2 = 'duipai.exe'

for i in range(1000000):

    tc = random.randint(1,10)

    # lee = random.randint(1, 2000)

    with open('test.data','w') as f:
        # f.write("1\n")
        f.write(f"{tc}\n")
        for ti in range(tc):
            nn = random.randint(1,30)
            kk = random.randint(1,15)
            f.write(f"{nn} {kk}\n")
            for ni in range(nn):
                elem = random.randint(1,1000)
                f.write(f"{elem} ")
            f.write('\n')
        # f.write('\n')
        # f.write('0\n')
        # for i in l2:
        #     f.write(f"{i} ")
        # f.write('\n')
        # f.write('1\n')
        # f.write(s1+'\n')
        # f.write(s2+'\n')
    # print('cpping')
    # os.system('duip < test.data > tcpp.out')
    res1 = os.popen(f'{exe1} < test.data').read().strip()
    # print('pying')
    # os.system('dd.py < test.data > tpy.out')
    res2 = os.popen(f'{exe2} < test.data').read().strip()
    # with open('tpy.out','r') as f:
    #     tpy = f.read()
    # with open('tcpp.out', 'r') as f:
    #     tcpp = f.read()
    if res1!=res2:
        # print(f'ERROR OCCURED at s1={s1} with s2={s2}')
        print(f"{exe1} ====>")
        print(res1)
        print('DIFFERS =====')
        print(res2)
        print(f"{exe2} <====")
        break
    if i%1000 == 0:
        print(f'testing {i+1}th data...')
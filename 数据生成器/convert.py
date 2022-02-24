import os, sys

d = 'out'

for i in os.listdir(d):
    fd = f"{d}/{i}"
    print(fd)
    with open(fd, 'r') as f:
        s = f.read().replace('\r','')
    with open(fd, 'w') as f:
        f.write(s)
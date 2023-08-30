

import os

# letter 1.60466
# covtype 146.284
# epsilon 3070.07
# gisette 114.718
# msd 180.434
# radar 231.193
# realsim 5784.3

datasets_size_mb = {
    'letter': 1.60466,
    'covtype': 146.284,
    'epsilon': 3070.07,
    'gisette': 114.718,
    'msd': 180.434,
    'radar': 231.193,
    'realsim': 5784.3
}

def foo(folder):
    os.chdir(folder)
    rx_begins = []
    rx_ends = []

    tx_begins = []
    tx_ends = []

    algo = folder.split('-')[0]
    algo = algo.lower()

    for i in range(4):
        with open(f"begin{i}.txt", "r") as f:
            begin_lines = f.readlines()
            for line in begin_lines:
                if 'bytes' in line:
                    line = line.strip()
                    if "RX" in line:
                        rx_begins.append(line.split('bytes ')[1].split(' ')[0])
                    else:
                        tx_begins.append(line.split('bytes ')[1].split(' ')[0])

        with open(f"end{i}.txt", "r") as f:
            end_lines = f.readlines()
            for line in end_lines:
                if 'bytes' in line:
                    line = line.strip()
                    if "RX" in line:
                        rx_ends.append(line.split('bytes ')[1].split(' ')[0])
                    else:
                        tx_ends.append(line.split('bytes ')[1].split(' ')[0])

    
    rx = f"{(int(rx_ends[0]) - int(rx_begins[0]))/1024/1024 - datasets_size_mb[algo]:.2f}"
    tx = f"{(int(tx_ends[0]) - int(tx_begins[0]))/1024/1024:.2f}"
    print('"%s": {"rx":%s, "tx":%s},' % (algo, rx, tx))

    # print(f"    party0 Total {((int(tx_ends[i]) - int(tx_begins[i])) + (int(rx_ends[i]) - int(rx_begins[i])))/1024/1024:.2f}MB")
    # calculate avg rx and tx
    rx_sum = 0
    tx_sum = 0
    for i in range(len(rx_begins)):
        print("Party %d RX: %.2f MB" % (i, (int(rx_ends[i]) - int(rx_begins[i]))/1024/1024))
        rx_sum += (int(rx_ends[i]) - int(rx_begins[i]))
        print("Party %d TX: %.2f MB" % (i, (int(tx_ends[i]) - int(tx_begins[i]))/1024/1024))
        tx_sum += (int(tx_ends[i]) - int(tx_begins[i]))
        print("")
    # print(f"Folder: {folder}")
    # print(f"RX avg: {rx_sum/1024/1024/len(rx_begins) - datasets_size_mb[algo]:.2f} MB")
    # print(f"TX avg: {tx_sum/1024/1024/len(tx_begins):.2f} MB")
    
    os.chdir('..')

print("{")
folders = os.listdir('.')
for folder in folders:
    if "proc" in folder:
        continue
    try:
        os.chdir(os.path.abspath(__file__)[:-7])
        foo(folder)
    except Exception as e:
        print(e)
        continue
print("}")
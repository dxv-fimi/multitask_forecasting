def main():
    pass
    # train_test_split()
    get_max()
    
def get_max():
    import numpy as np
    import pandas as pd
    data = pd.read_csv('./dataset/precipitation/data.csv').to_numpy()
    l = []
    for d in data:
        l += list(d)
    l = list(set(l))
    l.sort(reverse=True)
    print(l[:10])
    print(min([l[i-1]-l[i] for i in range(1, len(l))]))

def train_test_split():
    import math
    with open('./dataset/precipitation/data.csv') as f:
        data = f.read().split('\n')

    train = math.floor(len(data) * 0.7)

    with open('./dataset/precipitation/data_train.csv', 'w') as f:
        f.write('\n'.join(data[:train]))

    with open('./dataset/precipitation/data_test.csv', 'w') as f:
        f.write('\n'.join(data[train:]))

if __name__ == '__main__':
    main()

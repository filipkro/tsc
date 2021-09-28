import numpy as np
from argparse import ArgumentParser
import pandas as pd


def main(args):
    info = pd.read_csv(args.info, delimiter=',')
    first_data = np.where(info.values[:, 0] == 'index')[0][0] + 1
    data = np.array(info.values[first_data:, :4], dtype=int)

    same = 1
    subjects = 1
    observed = []
    unique = 0
    for i in range(1,data.shape[0]):
        if info.values[i+first_data,5] == info.values[i+first_data-1,5]:
            if not data[i,3] in observed:
                observed.append(data[i,3])
                # print('aooendsesd')
                # print(data[i,3])
        else:
            # print('len',len(observed))
            unique += len(observed)
            observed = []

        if  (data[i,3] != data[i-1,3]) or (info.values[i+first_data, 4] != info.values[i+first_data-1, 4]) or (info.values[i+first_data, 5] != info.values[i+first_data-1, 5]) or (data[i,1] != data[i-1,1]):
            same +=1

        # if data[i,1] == data[i-1,1]:
        #     same += 1
        #     if same > 5:
        #         print(same)
        #         print(info.values[first_data + i,:])
        # else:
        #     same = 0
    unique += len(observed)
    print(unique)
    print(same)
    print(data.shape[0])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('info')
    main(parser.parse_args())

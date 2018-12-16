import pandas as pd
import numpy as np

def load_dataset():
    train_dataset = pd.read_csv('./datas/train.csv',parse_dates=True,usecols=range(1,95))
    #train_dataset = train_dataset.values
    train_dataset = np.array(train_dataset)

    train_set_x = train_dataset[:,0:93]
    train_set_y = train_dataset[:,93:94]

    test_set_x = pd.read_csv('./datas/test.csv',parse_dates=True,usecols=range(1,94))
    test_set_x = np.array(test_set_x)
    
    train_set_y = convertstr2num(train_set_y)

    test_num = test_set_x.shape[0]
    test_set_y = [1]*test_num
    test_set_y = np.array(test_set_y)

    return train_set_x,train_set_y,test_set_x,test_set_y

def convertstr2num(matrix):
    matrix = np.array(matrix)
    data_set = []
    for str_rows in matrix:
        str_rows = str(str_rows[0])
        str_matrix = str_rows.split('_',1)[1]
        
        data_set.append(int(str_matrix))    
    data_set = np.array(data_set)
    return data_set


if __name__=='__main__':
    train_set_x,train_set_y,test_set_x,test_set_y = load_dataset();
    print(train_set_x.shape)
    print(train_set_x)
    print(train_set_y.shape)
    print(train_set_y)
    
    print(test_set_x.shape)
    print(test_set_x)
    print(test_set_y.shape)
    print(test_set_y)
    
    

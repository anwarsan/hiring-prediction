import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def data_cleaning(df):
    df = df.drop(columns=['date','hair'])
    df = df[df.age > 14]
    df = df[df.exp > -1 ]
    df = df[df.note <= 100]
    #print('data cleaning set size:', df.shape)
    return df


def data_encoding(df):
    #  encoding
    le = LabelEncoder()
    
    # availability and gender are binaries
    df = df.copy()
    df['gender'] = le.fit_transform(df['gender'])
    # 0: F , 1: M
    df['availability'] = le.fit_transform(df['availability'])
    # 0: no, 1: yes
    
    # one-hot encoding for features
    df = pd.get_dummies(df)    
    return df


def data_splitting(df):
    X = df.drop(['hiring'], axis =1)
    y = df['hiring'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0) # unbalanced => stratify
   
    return X_train, X_test, y_train, y_test

def data_scaling(train, test):
    std_scale = StandardScaler()
    std_scale.fit(train)
    X_train_std = std_scale.transform(train)
    X_test_std = std_scale.transform(test)
        
    return X_train_std, X_test_std

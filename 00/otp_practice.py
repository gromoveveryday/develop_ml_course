import pandas as pd
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from optbinning import BinningProcess
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix

test = pd.read_excel(r'Z:\DATASETS\otp\test.xls')
train = pd.read_excel(r'Z:\DATASETS\otp\train.xls')
y_train = train['TARGET']
y_test = pd.read_excel(r'Z:\DATASETS\otp\submission.xls')
test = pd.concat([test, y_test], axis=1)

def optimal_model(train, test, n_bins, target_name, y_train, y_test):
    train_clean = train.drop(['AGREEMENT_RK', 'GEN_INDUSTRY', 'GEN_TITLE', 'ORG_TP_STATE', 'ORG_TP_FCAPITAL'], axis=1) # По результату анализа убираем ненужные столбцы 
    test_clean = test.drop(['AGREEMENT_RK', 'GEN_INDUSTRY', 'GEN_TITLE', 'ORG_TP_STATE', 'ORG_TP_FCAPITAL'], axis=1)

    for column in train_clean.columns: # Замена пустых объектов на '0' и пустых целочисленных/вещественных значений на 0 
        if train_clean[column].dtype == 'object':
            train_clean[column] = train_clean[column].fillna('0')
        elif train_clean[column].dtype in ['int64', 'float64']:
            train_clean[column] = train_clean[column].fillna(0)
    
    for column in test_clean.columns:
        if test_clean[column].dtype == 'object':
            test_clean[column] = test_clean[column].fillna('0')
        elif test_clean[column].dtype in ['int64', 'float64']:
            test_clean[column] = test_clean[column].fillna(0)
    
    train_clean['PREVIOUS_CARD_NUM_UTILIZED'] = train_clean['PREVIOUS_CARD_NUM_UTILIZED'].astype('object') # Столбец PREVIOUS_CARD_NUM_UTILIZED не является объектом, нужно менянять в обоих фреймах
    test_clean['PREVIOUS_CARD_NUM_UTILIZED'] = test_clean['PREVIOUS_CARD_NUM_UTILIZED'].astype('object')

    categories = ['SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'GENDER', 'EDUCATION', 'MARITAL_STATUS', 
              'JOB_DIR', 
              'REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE',
              'TP_PROVINCE', 'REGION_NM', 'REG_FACT_FL', 'FACT_POST_FL', 'REG_POST_FL',
              'REG_FACT_POST_FL', 'REG_FACT_POST_TP_FL', 'FL_PRESENCE_FL', 'OWN_AUTO',
              'AUTO_RUS_FL', 'HS_PRESENCE_FL', 'COT_PRESENCE_FL', 'GAR_PRESENCE_FL',
              'LAND_PRESENCE_FL', 'DL_DOCUMENT_FL', 'GPF_DOCUMENT_FL', 'FACT_PHONE_FL',
              'REG_PHONE_FL', 'GEN_PHONE_FL', 'PREVIOUS_CARD_NUM_UTILIZED'] # Список столбцов-характеристик
    
    values = ['AGE', 'CHILD_TOTAL', 'DEPENDANTS', 'FAMILY_INCOME', 'PERSONAL_INCOME', 'CREDIT',
          'TERM', 'FST_PAYMENT', 'FACT_LIVING_TERM', 'WORK_TIME', 'LOAN_NUM_TOTAL',
          'LOAN_NUM_CLOSED', 'LOAN_NUM_PAYM', 'LOAN_DLQ_NUM', 'LOAN_MAX_DLQ',
          'LOAN_AVG_DLQ_AMT', 'LOAN_MAX_DLQ_AMT'] # Список столбцов-значений
    
    for category_name in values: # Оптимальное квантование
         
         a = list()
         a.append(category_name) 
         binning_process_1 = BinningProcess(a, max_n_bins=n_bins)
         x1_1 = train_clean[a].values
         x2_1 = train_clean[target_name]
         binning_process_1.fit(x1_1, x2_1)
         train_clean[category_name] = binning_process_1.transform(x1_1, metric='woe')
    encoder_1 = ce.OneHotEncoder(cols=categories)
    encoded_train = encoder_1.fit_transform(train_clean)

    for category_name in values:
         b = list()
         b.append(category_name) 
         binning_process_2 = BinningProcess(b, max_n_bins=n_bins)
         x1_2 = test_clean[b].values
         x2_2 = test_clean[target_name]
         binning_process_2.fit(x1_2, x2_2)
         test_clean[category_name] = binning_process_2.transform(x1_2, metric='woe')
    encoder_2 = ce.OneHotEncoder(cols=categories)
    encoded_test = encoder_2.fit_transform(test_clean)

    encoded_train = encoded_train.drop([target_name], axis=1)
    encoded_test = encoded_test.drop([target_name, 'POSTAL_ADDRESS_PROVINCE_81'], axis=1) # В проверочных данных новый столбец, которого нет в обучающей выборке

    log_reg = LogisticRegression(
        penalty='l1', # Штраф - L1
        dual=False, # Для другой размерности данных
        tol=1e-5, # погрешность остановки алгоритма
        C=0.03, # Обратное значение параметра регуляризации
        fit_intercept=False, # есть ли b0 (свободный член)
        intercept_scaling=1, # масшабирование b0 в случае L1
        class_weight='balanced', # 
        solver='liblinear', # алгоритм оптимизации 
        max_iter=100, # максимальное количество итераций оптимизации
        multi_class='auto', # 
        verbose=1, # подробный вывод? 
        warm_start=True, # использоватать предыдущие значения для переобучения 
        n_jobs=-1, # количество потоков для параллельного выполнения 
        random_state=42 # контроль случайности
    )
    
    log_reg.fit(encoded_train, y_train)
    
    pred_4 = log_reg.predict(encoded_test)
    y_pred_proba_4 = log_reg.predict_proba(encoded_test)
    roc_auc_4 = roc_auc_score(y_test, y_pred_proba_4[:,1])
    
    return 'AUC модели Лог рег: ', roc_auc_4

optimal_model(train=train, test=test, n_bins=100, target_name='TARGET', y_train=y_train, y_test=y_test)
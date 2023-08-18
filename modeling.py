import pandas as pd
import numpy as np
import pycaret
from pycaret.classification import *
import os
import datetime
from sklearn.preprocessing import StandardScaler

def my_model(args):
    
    if not os.path.exists(args.model_weight_path):
        os.makedirs(args.model_weight_path)
    if not os.path.exists(args.predict_result_path):
        os.makedirs(args.predict_result_path)
    #
    data = pd.read_csv(f"/data/Stock_US/prepro/{args.data}_{args.time}h.csv").iloc[:, 1:]
    data["Date"] = data["Date"].astype(str) +" "+ data["Time"]
    data["Date"] = pd.to_datetime(data["Date"])

    input_columns = ["Date",'z_open', 'z_high', 'z_low', 'z_close', 'z_d5', 'z_d10', 'z_d15','z_d20', 'z_d25', 'z_d30', 'rsi', 'macd', 'v_macd', 'natr']
    target_columns = [f'{args.target_col}']

    data = data[input_columns + target_columns]

    train = data[data["Date"] <= "2021-06-20"]
    test = data[(data["Date"] > "2021-06-20")]
    
    if args.date == True:
        print("DATE : 1")
        pass
    else:
        print("DATE : 0")
        train_date = train[["Date"]]
        test_date = test[["Date"]]
        train = train.drop("Date", axis = 1)
        test = test.drop("Date", axis = 1)
    
    std_scaler = StandardScaler()
    tech_fitted_ = std_scaler.fit(train[['rsi', 'macd', 'v_macd', 'natr']])
    transf_train =  pd.DataFrame(std_scaler.transform(train[['rsi', 'macd', 'v_macd', 'natr']]))
    transf_train.columns = ['rsi', 'macd', 'v_macd', 'natr']
    
    transf_test = pd.DataFrame(std_scaler.transform(test[['rsi', 'macd', 'v_macd', 'natr']]))
    transf_test.columns = ['rsi', 'macd', 'v_macd', 'natr']
    
    if args.norm == True:
        print("NORM : 1")
        
        if args.date == True:
            train = train.drop(['rsi', 'macd', 'v_macd', 'natr'], axis = 1).reset_index().iloc[:, 1:]
            train = pd.concat([train, transf_train], axis = 1)
            train = train[["Date",'z_open', 'z_high', 'z_low', 'z_close', 'z_d5', 'z_d10', 'z_d15','z_d20', 'z_d25', 'z_d30', 'rsi', 'macd', 'v_macd', 'natr', target_columns[0]]]

            test = test.drop(['rsi', 'macd', 'v_macd', 'natr'], axis = 1).reset_index().iloc[:, 1:]
            test = pd.concat([test, transf_test], axis = 1)
            test = test[["Date",'z_open', 'z_high', 'z_low', 'z_close', 'z_d5', 'z_d10', 'z_d15','z_d20', 'z_d25', 'z_d30', 'rsi', 'macd', 'v_macd', 'natr', target_columns[0]]]    
        else:
            train = train.drop(['rsi', 'macd', 'v_macd', 'natr'], axis = 1).reset_index().iloc[:, 1:]
            train = pd.concat([train, transf_train], axis = 1)
            train = train[['z_open', 'z_high', 'z_low', 'z_close', 'z_d5', 'z_d10', 'z_d15','z_d20', 'z_d25', 'z_d30', 'rsi', 'macd', 'v_macd', 'natr', target_columns[0]]]

            test = test.drop(['rsi', 'macd', 'v_macd', 'natr'], axis = 1).reset_index().iloc[:, 1:]
            test = pd.concat([test, transf_test], axis = 1)
            test = test[['z_open', 'z_high', 'z_low', 'z_close', 'z_d5', 'z_d10', 'z_d15','z_d20', 'z_d25', 'z_d30', 'rsi', 'macd', 'v_macd', 'natr', target_columns[0]]]    
    
    else:
        print("NORM : 0")
        pass


    exp_btc_tl = setup(train, target = target_columns[0], session_id = 111, fold = 3, use_gpu = True, silent = True, verbose = False)
    print("")
    
    
    svm_ = create_model('svm')
    rf_ = create_model('rf')
    knn_ = create_model('knn')
    mlp_ = create_model('mlp')
    catboost_ = create_model('catboost')
    xgboost_ = create_model('xgboost')

    et_ = create_model('et')

    tuned_svm_ = tune_model(svm_, fold = 3 , optimize ="AUC",  search_library= "optuna",choose_better = True)
    tuned_xgboost_ = tune_model(xgboost_, fold = 3, optimize ="AUC",  search_library= "optuna",choose_better = True)
    tuned_rf_ = tune_model(rf_, fold = 3, optimize = "AUC", search_library= "optuna",choose_better = True)
    tuned_knn_ = tune_model(knn_, fold = 3, optimize = "AUC", search_library= "optuna",choose_better = True)
    
    tuned_mlp_ = tune_model(mlp_, fold = 3, optimize = "AUC", search_library= "scikit-learn",choose_better = True)
    tuned_catboost_ = tune_model(catboost_, fold = 3, optimize = "AUC", search_library= "optuna",choose_better = True)
    tuned_et_      = tune_model(et_, fold = 3, optimize = "AUC", search_library= "optuna",choose_better = True )

    
    final_svm = finalize_model(tuned_svm_)
    final_xg = finalize_model(tuned_xgboost_)
    final_rf = finalize_model(tuned_rf_)
    final_knn = finalize_model(tuned_knn_)
    
    final_mlp = finalize_model(tuned_mlp_)
    final_cat = finalize_model(tuned_catboost_)
    final_et = finalize_model(tuned_et_)

    final_result_a =  predict_model(final_svm, data = test)
    final_result_b =  predict_model(final_xg, data = test)
    final_result_c =  predict_model(final_rf, data = test)
    final_result_d =  predict_model(final_knn, data = test)
    final_result_e =  predict_model(final_mlp, data = test)
    final_result_f =  predict_model(final_cat, data = test)
    final_result_g =  predict_model(final_et, data = test)

    #save_model(final_rf,f'./{args.model_weight_path}/{args.data}_{args.target_col}_rf')
    #save_model(final_svm,f'./{args.model_weight_path}/{args.data}_{args.target_col}_svm')
    #save_model(final_xg,f'./{args.model_weight_path}/{args.data}_{args.target_col}_xg')
    #save_model(final_knn,f'./{args.model_weight_path}/{args.data}_{args.target_col}_knn')
    #save_model(final_mlp,f'./{args.model_weight_path}/{args.data}_{args.target_col}_mlp')
    #save_model(final_cat,f'./{args.model_weight_path}/{args.data}_{args.target_col}_cat')
    #save_model(final_et,f'./{args.model_weight_path}/{args.data}_{args.target_col}_et')

    final_result_a.to_csv(f"./{args.predict_result_path}/{args.data}_{args.time}h_{args.target_col}_svm.csv") 
    final_result_b.to_csv(f"./{args.predict_result_path}/{args.data}_{args.time}h_{args.target_col}_xg.csv") 
    final_result_c.to_csv(f"./{args.predict_result_path}/{args.data}_{args.time}h_{args.target_col}_rf.csv") 
    final_result_d.to_csv(f"./{args.predict_result_path}/{args.data}_{args.time}h_{args.target_col}_knn.csv") 
    final_result_e.to_csv(f"./{args.predict_result_path}/{args.data}_{args.time}h_{args.target_col}_mlp.csv")
    final_result_f.to_csv(f"./{args.predict_result_path}/{args.data}_{args.time}h_{args.target_col}_cat.csv") 
    final_result_g.to_csv(f"./{args.predict_result_path}/{args.data}_{args.time}h_{args.target_col}_et.csv") 
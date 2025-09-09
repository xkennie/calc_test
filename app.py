import pandas as pd
#import joblib
import lightgbm as lgb
import pickle
import numpy as np
import category_encoders as ce
import streamlit as st
from category_encoders.one_hot import OneHotEncoder

def calculator_retention(df, #data 
                        eta, #ETA
                        cte, #СТЕ (оставить распределение из выборки, можно 86ести руками)
                        total_cost, #Стоимость доставки
                        cancel_share, #Доля отмен
                        late_min, #Опоздания, в минутах
                        late_share,
                        retailer_category,#Доля заказов с опозданием
                        model,#model+encoder
                        ohe):
    data = df.copy()
    log_cancel = cancel_share*0.6
    other_cancel = cancel_share*0.4
    no_cancel = 1-cancel_share
    if eta > -1:
        data["right_eta"] = eta
    if cte > -1:
        data["duration_click_to_eat"] = cte
    if total_cost > -1:    
        data["total_cost"] = total_cost/100
    if cancel_share > -1: 
        data["cancel_tag"] = np.random.choice([' ', 'Логистические отмены', 'Прочее'],
                                                size=data.shape[0], p=[no_cancel,log_cancel,other_cancel])
    if late_share > -1:
        data["late"] = np.random.choice([0, late_min],
                                                size=data.shape[0], p=[1-late_share,late_share])


    #X_test_new['repeated_or_new'] = np.random.choice(['repeated', 'new'], size=1000000, p=[0.97,0.03])

    data['device_type'] = np.random.choice(['mobile', 'desktop'],
                                                size=data.shape[0], p=[0.96,0.04])

    data['prime_flg'] = np.random.choice([1, 0],
                                                size=data.shape[0], p=[0.43,0.57])

    data['loyal_user_flg'] = np.random.choice(['loyal', 'not_loyal'],
                                                size=data.shape[0], p=[0.66,0.34])

    data['os'] = np.random.choice(['android', 'ios', 'windows'],
                                                size=data.shape[0], p=[0.54, 0.41, 0.05])
    if retailer_category == 'all':
      data['retailer_category_name'] = np.random.choice(['grocery', 'rte', 'other'],
                                                  size=data.shape[0], p=[0.7,0.21, 0.09])
      data['retailer_name'] = np.random.choice(['пятерочка', 'магнит', 'перекресток', 'вкусно - и точка', "rostic's", 'Прочее'],
                                                size=data.shape[0], p=[0.33,0.18, 0.09, 0.07, 0.06, 0.27])

    elif retailer_category == 'rte':
      data['retailer_category_name'] = retailer_category
      data['retailer_name'] = np.random.choice(['вкусно - и точка', "rostic's", 'Прочее'],
                                                size=data.shape[0], p=[0.175, 0.15, 0.675])
    elif retailer_category == 'grocery':
      data['retailer_category_name'] = retailer_category
      data['retailer_name'] = np.random.choice(['пятерочка', 'магнит', 'перекресток', 'Прочее'],
                                                size=data.shape[0], p=[0.39, 0.2, 0.1, 0.31])
    


    data['retailer_name'] = np.random.choice(['пятерочка', 'магнит', 'перекресток', 'вкусно - и точка', "rostic's", 'Прочее'],
                                                size=data.shape[0], p=[0.33,0.18, 0.09, 0.07, 0.06, 0.27])

    data["tenant_id"] = 'sbermarket'

    data = ohe.transform(data)

    y_test_pred = model.predict(data)
    if total_cost == 0:
      retens = round(100*y_test_pred.mean()*(1-0.303), 1) #срезается конверсии до заказа
    elif total_cost > 0 and total_cost <= 50:
      retens = round(100*y_test_pred.mean()*(1+0.084), 1)
    elif total_cost > 50 and total_cost <= 100:
      retens = round(100*y_test_pred.mean()*(1+0.073), 1)
    elif total_cost > 100 and total_cost <= 150:
      retens = round(100*y_test_pred.mean()*(1+0.085), 1)
    elif total_cost > 150 and total_cost <= 200:
      retens = round(100*y_test_pred.mean()*(1-0.043), 1)
    elif total_cost > 200 and total_cost <= 300:
      retens = round(100*y_test_pred.mean()*(1-0.089), 1)
    elif total_cost > 300 and total_cost <= 400:
      retens = round(100*y_test_pred.mean()*(1-0.222), 1)
    elif total_cost > 400:
      retens = round(100*y_test_pred.mean()*(1-0.288), 1)
    else:
      retens = round(100*y_test_pred.mean(), 1)
    print(f"Ожидаемый ретеншн при заданных вводных: {retens}%")
    
    orders = 1600000
    SHp = 33
    CAC_cour = 5300
    churn_rate = 0.55
    util = 0.75

    if eta > 0 and cte == 0:
        cpo_cte = (1-late_share)*eta+late_share*(eta+late_min)
    else:
        cpo_cte = cte
    
    sh_needed = orders*(1-cancel_share)*(cpo_cte/util)/60
    hire_costs = CAC_cour*churn_rate*(sh_needed/SHp)
    hire_cpo = hire_costs/orders
    
    if total_cost > 0:
        total_cpo = hire_cpo+total_cost
    else:
        total_cpo = hire_cpo+round(data.total_cost.mean()*100, 0) 
    
    print(f"Ожидаемый CPO системы: {round(total_cpo, 0)}₽")
    return retens, total_cpo
    

#warnings.filterwarnings('ignore')
#lgb_model = joblib.load('lgb.pkl')
with open('new_lgb.pkl', 'rb') as f: #changed model
    lgb_model = pickle.load(f)
with open('one_hot_encoder.pkl', 'rb') as f:
    ohe = pickle.load(f)
    
st.set_page_config(page_title="Retention&CPO Calculator", layout="wide")

st.title("Calculator")
st.markdown("**Примерный калькулятор ретеншна и CPO от метрик**")

st.markdown("---")

st.write("Загрузите датасет")
uploaded_file = st.file_uploader("Выберите файл (CSV или Excel)", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    # Определяем тип файла по расширению
    file_name = uploaded_file.name.lower()

    df = pd.read_csv(uploaded_file, sep=",")

    #st.write("Загруженный набор данных:")
    #st.dataframe(df)
    col1, col2 = st.columns(2)
    with col1:
      category_setting_c1 = st.text_input("Категория ритейлера", value = "rte", key = 'category_main')
      eta_setting_c1 = st.text_input("Таргет правой границы ЕТА", value = 35, key = 'eta_main')
      cte_setting_c1 = st.text_input("Таргет CTE", value = 40, key = 'cte_main')
      late_minutes_setting_c1 = st.text_input("Таргет минут опоздания", value = 10, key = 'late_minutes_main')
      late_share_setting_c1 = st.text_input("Таргет доли опозданий", value = 0.1, key = 'late_share_main')
      total_cost_setting_c1 = st.text_input("Таргет стоимости доставки", value = 100, key = 'total_cost_main')
      cancel_share_setting_c1 = st.text_input("Таргет доли отмен", value = 0.05, key = 'cancel_share_main')
      
      if st.checkbox("Выполнить расчёт: основной"):
        r1, c1 = calculator_retention(df = df, #data 
                          eta = eval(eta_setting_c1), #ETA
                          cte = eval(cte_setting_c1), #СТЕ (оставить распределение из выборки, можно 86ести руками)
                          total_cost = eval(total_cost_setting_c1), #Стоимость доставки
                          cancel_share = eval(cancel_share_setting_c1), #Доля отмен
                          late_min = eval(late_minutes_setting_c1), #Опоздания, в минутах
                          late_share = eval(late_share_setting_c1), 
                          retailer_category = category_setting_c1,
                          model = lgb_model,#model+encoder
                          ohe = ohe)
        st.write(f"Ожидаемый ретеншн при заданных вводных: {r1}%")
        st.write(f"Ожидаемый CPO системы: {round(c1, 1)}₽")
        st.write(f"*CPO считается в тестовом режиме")
    with col2:
      category_setting_c2 = st.text_input("Категория ритейлера", value = "grocery", key = 'category_comp')  
      eta_setting_c2 = st.text_input("Таргет правой границы ЕТА", value = 30, key = 'eta_comp')
      cte_setting_c2 = st.text_input("Таргет CTE", value = 35, key = 'cte_comp')
      late_minutes_setting_c2 = st.text_input("Таргет минут опоздания", value = 15, key = 'late_minutes_comp')
      late_share_setting_c2 = st.text_input("Таргет доли опозданий", value = 0.05, key = 'late_share_comp')
      total_cost_setting_c2 = st.text_input("Таргет стоимости доставки", value = 70, key = 'total_cost_comp')
      cancel_share_setting_c2 = st.text_input("Таргет доли отмен", value = 0.07, key = 'cancel_share_comp')

      if st.checkbox("Выполнить расчёт: сравнение"):
        r2, c2 = calculator_retention(df = df, #data 
                          eta = eval(eta_setting_c2), #ETA
                          cte = eval(cte_setting_c2), #СТЕ (оставить распределение из выборки, можно 86ести руками)
                          total_cost = eval(total_cost_setting_c2), #Стоимость доставки
                          cancel_share = eval(cancel_share_setting_c2), #Доля отмен
                          late_min = eval(late_minutes_setting_c2), #Опоздания, в минутах
                          late_share = eval(late_share_setting_c2),
                          retailer_category = category_setting_c2,
                          model = lgb_model,#model+encoder
                          ohe = ohe)
        st.write(f"Ожидаемый ретеншн при заданных вводных: {r2}%")
        st.write(f"Ожидаемый CPO системы: {round(c2, 1)}₽")
        st.write(f"*CPO считается в тестовом режиме")


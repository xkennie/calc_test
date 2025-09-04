import pandas as pd
#import joblib
import lightgbm as lgb
import pickle
import numpy as np
import category_encoders as ce
import streamlit as st
import lightgbm as lgb
from category_encoders.one_hot import OneHotEncoder

def calculator_retention(df, #data 
                        eta, #ETA
                        cte, #СТЕ (оставить распределение из выборки, можно 86ести руками)
                        total_cost, #Стоимость доставки
                        cancel_share, #Доля отмен
                        late_min, #Опоздания, в минутах
                        late_share, #Доля заказов с опозданием
                        model,#model+encoder
                        ohe):
    data = df.copy()
    log_cancel = cancel_share*0.6
    other_cancel = cancel_share*0.4
    no_cancel = 1-cancel_share
    if eta > 0:
        data["right_eta"] = eta
    if cte > 0:
        data["duration_click_to_eat"] = cte
    if total_cost > 0:    
        data["total_cost"] = total_cost/100
    if cancel_share > 0: 
        data["cancel_tag"] = np.random.choice([' ', 'Логистические отмены', 'Прочее'],
                                                size=1000000, p=[no_cancel,log_cancel,other_cancel])
    if late_share > 0:
        data["late"] = np.random.choice([0, late_min],
                                                size=1000000, p=[1-late_share,late_share])


    #X_test_new['repeated_or_new'] = np.random.choice(['repeated', 'new'], size=1000000, p=[0.97,0.03])

    data['device_type'] = np.random.choice(['mobile', 'desktop'],
                                                size=1000000, p=[0.96,0.04])

    data['prime_flg'] = np.random.choice([1, 0],
                                                size=1000000, p=[0.43,0.57])

    data['loyal_user_flg'] = np.random.choice(['loyal', 'not_loyal'],
                                                size=1000000, p=[0.66,0.34])

    data['os'] = np.random.choice(['android', 'ios', 'windows'],
                                                size=1000000, p=[0.54, 0.41, 0.05])

    data['retailer_category_name'] = np.random.choice(['grocery', 'rte', 'other'],
                                                size=1000000, p=[0.7,0.21, 0.09])

    data['retailer_name'] = np.random.choice(['пятерочка', 'магнит', 'перекресток', 'вкусно - и точка', "rostic's", 'Прочее'],
                                                size=1000000, p=[0.33,0.18, 0.09, 0.07, 0.06, 0.27])

    data["tenant_id"] = 'sbermarket'

    data = ohe.transform(data)

    y_test_pred = model.predict(data)
    retens = round(100*y_test_pred.mean(), 1)
    print(f"Ожидаемый ретеншн при заданных вводных: {round(100*y_test_pred.mean(), 1)}%")
    
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
with open('lgb.pkl', 'rb') as f:
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

    if file_name.endswith('.csv'):
        # Настройки для CSV файлов
        sep_sign = st.selectbox(
            "Выберите разделитель",
            (";", ",", " ", "|"), index=0)

        decimal_sign = st.selectbox(
            "Выберите отделитель дробной части",
            (".", ","), index=1)

        df = pd.read_csv(uploaded_file, sep=sep_sign, decimal=decimal_sign)

    elif file_name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)

    st.write("Загруженный набор данных:")
    st.dataframe(df)

    eta_setting = st.text_input("Таргет правой границы ЕТА", value = 35)
    cte_setting = st.text_input("Таргет CTE", value = 40)
    late_minutes_setting = st.text_input("Таргет минут опоздания", value = 10)
    late_share_setting = st.text_input("Таргет доли опозданий", value = 0.1)
    total_cost_setting = st.text_input("Таргет стоимости доставки", value = 100)
    cancel_share_setting = st.text_input("Таргет доли отмен", value = 0.05)
    
    if st.button("Выполнить расчёт"):
      r, c = calculator_retention(df = df, #data 
                        eta = eta_setting, #ETA
                        cte = cte_setting, #СТЕ (оставить распределение из выборки, можно 86ести руками)
                        total_cost = total_cost_setting, #Стоимость доставки
                        cancel_share = cancel_share_setting, #Доля отмен
                        late_min = late_min_setting, #Опоздания, в минутах
                        late_share = late_share_setting, #Доля заказов с опозданием
                        model = lgb_model,#model+encoder
                        ohe = ohe)
      st.write(f"Ожидаемый ретеншн при заданных вводных: {r}%")
      st.write(f"Ожидаемый CPO системы: {c}₽")

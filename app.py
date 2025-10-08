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
                        cte, #СТЕ (оставить распределение из выборки, можно ввести руками)
                        total_cost, #Стоимость доставки
                        cancel_share, #Доля отмен
                        late_min, #Опоздания, в минутах
                        late_share, #Доля заказов с опозданием
                        retailer_category, #Вертикаль
                        model,#model+encoder
                        ohe,
                        city):
    data = df.copy()
    if city != 'Страна': #filter city
        data = data[data['city_name'] == city]
        
    data = data[['morning_flg', 'evening_flg', 'new_or_repeated', 'retailer_name', 'device_type', 
              'os', 'rate', 'tenant_id', 'b2b', 'prime_flg', 'loyal_user_flg', 'retailer_category_name',
              'logcancel_flg', 'cancel_tag', 'surge_pay', 'duration_click_to_eat', 'spasibo_used',
              'promo_used', 'total_quantity', 'total', 'total_cost', 'replaced_items_cnt', 'canceled_items_cnt',
              'distance_to_store', 'isz_flg', 'right_eta', 'late', 'early', 'order_number', 'city', 'fee_share']] 
    np.random.seed(42)
    log_cancel = cancel_share*0.6 #раскидываю типы отмен по распределению
    other_cancel = cancel_share*0.4
    no_cancel = 1-cancel_share
    environment_setting = {'Омск': {'orders': 26348, 'users': 14023, 'sh_p': 22.12028518997218, 'mph': 302.98568417301266, 'cac': 14200},
                           'Страна': {'orders': 1600000, 'users': 760000, 'sh_p': 30, 'mph': 320, 'cac': 7000},
                           'Самара': {'orders': 28325, 'users': 15429, 'sh_p': 25.54027528722855, 'mph': 286.17744949793234, 'cac': 17500}, 'Воронеж': {'orders': 15760, 'users': 9119, 'sh_p': 33.246466484074496, 'mph': 332.8099072752656, 'cac': 13400}, 'Красноярск': {'orders': 42203, 'users': 20382, 'sh_p': 25.67888789400519, 'mph': 319.4503727831454, 'cac': 14900}, 'Краснодар': {'orders': 53861, 'users': 26180, 'sh_p': 36.17460198082647, 'mph': 279.61018651460904, 'cac': 9000}, 'Уфа': {'orders': 25785, 'users': 14319, 'sh_p': 19.430849565206966, 'mph': 230.81337674648051, 'cac': 20000}, 'Новосибирск': {'orders': 59615, 'users': 28590, 'sh_p': 28.144976899831402, 'mph': 299.2014640481861, 'cac': 16100}, 'Тюмень': {'orders': 39822, 'users': 17081, 'sh_p': 54.21273889158505, 'mph': 308.26935993788135, 'cac': 3700}, 'Ростов-на-Дону': {'orders': 36557, 'users': 19145, 'sh_p': 37.94114273226561, 'mph': 307.580050690035, 'cac': 14300}, 'Екатеринбург': {'orders': 60587, 'users': 30017, 'sh_p': 29.628387084910294, 'mph': 296.44725135645535, 'cac': 22548}, 'Санкт-Петербург': {'orders': 139257, 'users': 65363, 'sh_p': 33.51606552207418, 'mph': 356.3062228931145, 'cac': 18200}, 'Челябинск': {'orders': 27802, 'users': 13824, 'sh_p': 17.208883637740662, 'mph': 263.0693713434905, 'cac': 16000}, 'Москва': {'orders': 355855, 'users': 149260, 'sh_p': 54.38130719093238, 'mph': 348.80240077034074, 'cac': 21200}, 'Пермь': {'orders': 23453, 'users': 12207, 'sh_p': 25.114133484255216, 'mph': 287.2839768858267, 'cac': 13700}, 'Нижний Новгород': {'orders': 16583, 'users': 10207, 'sh_p': 27.815011220147408, 'mph': 300.26184460801244, 'cac': 13100}, 'Волгоград': {'orders': 21793, 'users': 10978, 'sh_p': 43.47757513021616, 'mph': 292.495519982006, 'cac': 29300}, 'Казань': {'orders': 31104, 'users': 17041, 'sh_p': 28.642809057633155, 'mph': 288.4276135448993, 'cac': 18600}}
    

    #хардом ставлю инпуты в выборку
    #даю возможность не указывать параметр, используя -1 -- тогда он останется по распределению в выборке
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


    #для корректного использования переменных категории/ритейлера генерирую их по распределению
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

    data["tenant_id"] = 'sbermarket'

    data = ohe.transform(data)

    #предикты для всей выборки
    y_test_pred = model.predict(data)
    
    #срезается конверсия на экране чекаута, до заказа
    #по сути заглушка для очень больших значений стоимости доставки, коэффициенты брал из ресёрча продуктовой команды
    if total_cost == 0:
        retens = round(100*y_test_pred.mean()*(1-0.303), 2) 
    #elif total_cost > 0 and total_cost <= 50:
    #    retens = round(100*y_test_pred.mean()*(1+0.084), 1)
    #elif total_cost > 50 and total_cost <= 100:
    #    retens = round(100*y_test_pred.mean()*(1+0.073), 1)
    #elif total_cost > 100 and total_cost <= 150:
    #    retens = round(100*y_test_pred.mean()*(1+0.085), 1)
    elif total_cost > 150 and total_cost <= 200:
        retens = round(100*y_test_pred.mean()*(1-0.043), 2)
    elif total_cost > 200 and total_cost <= 300:
        retens = round(100*y_test_pred.mean()*(1-0.089), 2)
    elif total_cost > 300 and total_cost <= 400:
        retens = round(100*y_test_pred.mean()*(1-0.222), 2)
    elif total_cost > 400:
        retens = round(100*y_test_pred.mean()*(1-0.288), 2)
    else:
        retens = round(100*y_test_pred.mean(), 2)
    #print(f"Ожидаемый ретеншн при заданных вводных: {retens}%")
    
    #Часть с CPO, неактуальная
    #orders = 1600000
    #SHp = 33
    #CAC_cour = 5300
    #churn_rate = 0.55
    #util = 0.75

    #if eta > 0 and cte == 0:
    #    cpo_cte = (1-late_share)*eta+late_share*(eta+late_min)
    #else:
    #    cpo_cte = cte
    
    #sh_needed = orders*(1-cancel_share)*(cpo_cte/util)/60
    #hire_costs = CAC_cour*churn_rate*(sh_needed/SHp)
    #hire_cpo = hire_costs/orders
    
    #if total_cost > 0:
    #    total_cpo = hire_cpo+total_cost
    #else:
    #    total_cpo = hire_cpo+round(data.total_cost.mean()*100, 0) 
   
    #часть с CPO
    orders = environment_setting[city]['orders']
    users = environment_setting[city]['users']
    SHp = environment_setting[city]['sh_p']
    newbie_rate = 0.42
    CAC_cour = environment_setting[city]['cac']
    aov = 8600
    MPH = environment_setting[city]['mph']
    
    if cte > -1:
        SH_per_order_need = 24/(0.003*(2.176*cte-75.7585) + 0.547)/60
        SH_need = SH_per_order_need * orders
        heads_need = SH_need / SHp
        heads_hire = heads_need * newbie_rate
        hire_costs = heads_hire * CAC_cour
        hire_cpo = hire_costs/orders
        
        
        direct_cpo = SH_per_order_need*MPH
        total_cpo = direct_cpo + hire_cpo
    elif eta > -1 and late_min > -1 and late_share > -1:
        cte_calc = eta+late_share*late_min
        
        SH_per_order_need = 24/(0.003*(2.176*cte_calc-75.7585) + 0.547)/60
        SH_need = SH_per_order_need * orders
        heads_need = SH_need / SHp
        heads_hire = heads_need * newbie_rate
        hire_costs = heads_hire * CAC_cour
        hire_cpo = hire_costs/orders
        
        direct_cpo = SH_per_order_need*MPH
        total_cpo = direct_cpo + hire_cpo
        
    elif cte <= -1 and eta <= -1:
        cte_calc = 45
        
        SH_per_order_need = 24/(0.003*(2.176*cte_calc-75.7585) + 0.547)/60
        SH_need = SH_per_order_need * orders
        heads_need = SH_need / SHp
        heads_hire = heads_need * newbie_rate
        hire_costs = heads_hire * CAC_cour
        hire_cpo = hire_costs/orders
        
        direct_cpo = SH_per_order_need*MPH
        total_cpo = direct_cpo + hire_cpo
    else:
        cte_calc = 45
        
        SH_per_order_need = 24/(0.003*(2.176*cte_calc-75.7585) + 0.547)/60
        SH_need = SH_per_order_need * orders
        heads_need = SH_need / SHp
        heads_hire = heads_need * newbie_rate
        hire_costs = heads_hire * CAC_cour
        hire_cpo = hire_costs/orders
        
        direct_cpo = SH_per_order_need*MPH
    
        total_cpo = direct_cpo + hire_cpo
        
    gmv = 0.01*retens*users*aov/1000000
    #print(f"Ожидаемая потребность в найме в CPO: {round(hire_cpo, 0)}₽")
    #print(f"Ожидаемый CPO директ: {round(direct_cpo, 0)}₽")    
    
    #print(f"Ожидаемый CPO системы: {round(total_cpo, 0)}₽")
    #print(f"Ожидаемый CPO системы: {round(total_cpo, 0)}₽")
    #возвращаем ожидаемый ретеншн системы и оценочный CPO
    return retens, hire_cpo, direct_cpo, total_cpo, gmv    

#warnings.filterwarnings('ignore')
#lgb_model = joblib.load('lgb.pkl')
with open('all_lgb.pkl', 'rb') as f: #changed model
    lgb_model = pickle.load(f)
with open('all_one_hot_encoder.pkl', 'rb') as f:
    ohe = pickle.load(f)

### Модели по городам
with open('voronezh_lgb.pkl', 'rb') as f: #changed model
    voronezh_model = pickle.load(f)
with open('volgograd_lgb.pkl', 'rb') as f: #changed model
    volgograd_model = pickle.load(f)
with open('ekaterinburg_lgb.pkl', 'rb') as f: #changed model
    ekaterinburg_model = pickle.load(f)
with open('kazan_lgb.pkl', 'rb') as f: #changed model
    kazan_model = pickle.load(f)
with open('krasnodar_lgb.pkl', 'rb') as f: #changed model
    krasnodar_model = pickle.load(f)
with open('krasnoyarsk_lgb.pkl', 'rb') as f: #changed model
    krasnoyarsk_model = pickle.load(f)
with open('moscow_lgb.pkl', 'rb') as f: #changed model
    moscow_model = pickle.load(f)
with open('nino_lgb.pkl', 'rb') as f: #changed model
    nino_model = pickle.load(f)
with open('novosibirsk_lgb.pkl', 'rb') as f: #changed model
    novosibirsk_model = pickle.load(f)
with open('omsk_lgb.pkl', 'rb') as f: #changed model
    omsk_model = pickle.load(f)
with open('perm_lgb.pkl', 'rb') as f: #changed model
    perm_model = pickle.load(f)
with open('rnd_lgb.pkl', 'rb') as f: #changed model
    rnd_model = pickle.load(f)
with open('samara_lgb.pkl', 'rb') as f: #changed model
    samara_model = pickle.load(f)
with open('spb_lgb.pkl', 'rb') as f: #changed model
    spb_model = pickle.load(f)
with open('tumen_lgb.pkl', 'rb') as f: #changed model
    tumen_model = pickle.load(f)
with open('ufa_lgb.pkl', 'rb') as f: #changed model
    ufa_model = pickle.load(f)
with open('chelyabinsk_lgb.pkl', 'rb') as f: #changed model
    chelyabinsk_model = pickle.load(f)

models = {
    'Волгоград': volgograd_model
    ,'Воронеж': voronezh_model
    ,'Екатеринбург': ekaterinburg_model 
    ,'Казань': kazan_model  
    ,'Краснодар': krasnodar_model
    ,'Красноярск': krasnoyarsk_model 
    ,'Москва': moscow_model
    ,'Нижний Новгород': nino_model 
    ,'Новосибирск': novosibirsk_model 
    ,'Омск': omsk_model
    ,'Пермь': perm_model
    ,'Ростов-на-Дону': rnd_model
    ,'Самара': samara_model 
    ,'Санкт-Петербург': spb_model 
    ,'Тюмень': tumen_model  
    ,'Уфа': ufa_model 
    ,'Челябинск': chelyabinsk_model
    ,'Страна': lgb_model
}
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
      #order_number_c1 = st.text_input("Номер заказа (-1 = не учитывать)", value = 15, key = 'order_main')
      city_c1 = st.text_input("Город (город топ-17 или Страна)", value = "Страна", key = 'city_main')
      category_setting_c1 = st.text_input("Категория ритейлера (all/rte/grocery)", value = "rte", key = 'category_main')
      eta_setting_c1 = st.text_input("Таргет правой границы ЕТА (25-90)", value = 35, key = 'eta_main')
      cte_setting_c1 = st.text_input("Таргет CTE (20-90)", value = 40, key = 'cte_main')
      late_minutes_setting_c1 = st.text_input("Таргет минут опоздания (0-30)", value = 10, key = 'late_minutes_main')
      late_share_setting_c1 = st.text_input("Таргет доли опозданий (0-0.25)", value = 0.1, key = 'late_share_main')
      total_cost_setting_c1 = st.text_input("Таргет стоимости доставки (0-500)", value = 100, key = 'total_cost_main')
      cancel_share_setting_c1 = st.text_input("Таргет доли отмен (0-0.3)", value = 0.05, key = 'cancel_share_main')
      
      if st.checkbox("Выполнить расчёт: основной"):

        r1, h1, d1, t1, g1 = calculator_retention(df = df, #data 
                          #order_number = eval(order_number_c1),
                          eta = eval(eta_setting_c1), #ETA
                          cte = eval(cte_setting_c1), #СТЕ (оставить распределение из выборки, можно ввести руками)
                          total_cost = eval(total_cost_setting_c1), #Стоимость доставки
                          cancel_share = eval(cancel_share_setting_c1), #Доля отмен
                          late_min = eval(late_minutes_setting_c1), #Опоздания, в минутах
                          late_share = eval(late_share_setting_c1), 
                          retailer_category = category_setting_c1,
                          model = models[city_c1],#model+encoder
                          ohe = ohe,
                          city = city_c1)
        st.write(f"Ожидаемый ретеншн при заданных вводных: {r1}%")
        st.write(f"Ожидаемый CPO системы: {round(t1, 1)}₽")
        st.write(f"Ожидаемый CPO директ: {round(d1, 1)}₽")
        st.write(f"Ожидаемый кост найма в CPO: {round(h1, 1)}₽")
        st.write(f"Ожидаемый GMV: {round(g1, 1)} млн ₽")
        #st.write(f"*CPO считается для страны")
    with col2:
      #order_number_c2 = st.text_input("Номер заказа (-1 = не учитывать)", value = 15, key = 'order_comp')
      city_c2 = st.text_input("Город (город топ-17 или Страна)", value = "Москва", key = 'city_comp')
      category_setting_c2 = st.text_input("Категория ритейлера (all/grocery/rte)", value = "grocery", key = 'category_comp')  
      eta_setting_c2 = st.text_input("Таргет правой границы ЕТА (25-90)", value = 30, key = 'eta_comp')
      cte_setting_c2 = st.text_input("Таргет CTE (20-90)", value = 35, key = 'cte_comp')
      late_minutes_setting_c2 = st.text_input("Таргет минут опоздания (0-30)", value = 15, key = 'late_minutes_comp')
      late_share_setting_c2 = st.text_input("Таргет доли опозданий (0-0.25)", value = 0.05, key = 'late_share_comp')
      total_cost_setting_c2 = st.text_input("Таргет стоимости доставки (0-500)", value = 70, key = 'total_cost_comp')
      cancel_share_setting_c2 = st.text_input("Таргет доли отмен (0-0.3)", value = 0.07, key = 'cancel_share_comp')

      if st.checkbox("Выполнить расчёт: сравнение"):
        r2, h2, d2, t2, g2 = calculator_retention(df = df, #data 
                          #order_number = eval(order_number_c2),
                          eta = eval(eta_setting_c2), #ETA
                          cte = eval(cte_setting_c2), #СТЕ (оставить распределение из выборки, можно ввести руками)
                          total_cost = eval(total_cost_setting_c2), #Стоимость доставки
                          cancel_share = eval(cancel_share_setting_c2), #Доля отмен
                          late_min = eval(late_minutes_setting_c2), #Опоздания, в минутах
                          late_share = eval(late_share_setting_c2),
                          retailer_category = category_setting_c2,
                          model = models[city_c2],#model+encoder
                          ohe = ohe,
                          city = city_c2)
        st.write(f"Ожидаемый ретеншн при заданных вводных: {r2}%")
        st.write(f"Ожидаемый CPO системы: {round(t2, 1)}₽")
        st.write(f"Ожидаемый CPO директ: {round(d2, 1)}₽")
        st.write(f"Ожидаемый кост найма в CPO: {round(h2, 1)}₽")
        st.write(f"Ожидаемый GMV: {round(g2, 1)} млн ₽")
        #st.write(f"*CPO считается для страны")

        

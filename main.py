#Miuul
#İngiltere merkezli perakende şirketi satış ve pazarlama
#faaliyetleri için roadmap belirlemek istemektedir. Şirketin
#orta uzun vadeli plan yapabilmesi için var olan müşterilerin
#gelecekte şirkete sağlayacakları potansiyel değerin
#tahmin edilmesi gerekmektedir

import pandas as pd
import numpy as np
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
import matplotlib.pyplot as plt
from lifetimes.plotting import plot_period_transactions

#Datasetini okumak
pd.set_option("display.max_columns",None)
pd.set_option("display.float_format",lambda x:'%.2f' %x)
df_=pd.read_excel("online_retail_II.xlsx",sheet_name="Year 2010-2011")
df=df_.copy()

#Dataset İncelemek
df.head()
df.isnull().sum()
df.dropna(inplace=True)
df.describe().T
df["Description"].nunique()
df.groupby("Invoice").agg({"TotalPrice":"sum"}).head()

#Veri Hazırlamak
df.dropna(inplace=True)
df=df[~df["Invoice"].str.contains("C",na=False)] #İptal edilen ürünler çıkartıldı
df["TotalPrice"]=df["Quantity"]*df["Price"]
df=df[df["Quantity"]>0]
df=df[df["Price"]>0]
today_date = dt.datetime(2011,12,25)




#Aykırı değerleri baskılamak adına gerekli fonksiyonlar:

def outlier_thresholds(dataframe,variable):
    quartile1=dataframe[variable].quantile(0.01)
    quartile3=dataframe[variable].quantile(0.99)
    interquartile_range =quartile3-quartile1
    up_limit=quartile3+1.5*interquartile_range
    low_limit=quartile1-1.5*interquartile_range
    return  low_limit,up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)


#Lifetime Veri Yapısının Hazırlanması
#recency:Son satın alma üstünden geçen zaman (Haftalık)
#T:Müşterinin yaşı (Haftalık)(Analiz tarihinden ne kadar önce satın alma yapılmış)
#frequency:Müşterinin satın alma sıklığı
#monetary_value:satın alma başına ortalama kazanç

cltv_df=df.groupby('Customer ID').agg({'InvoiceDate':[lambda date:(date.max()-date.min()).days,
                                                      lambda date:(today_date-date.min()).days],
                                       'Invoice':lambda num:num.nunique(),
                                       'TotalPrice':lambda TotalPrice:TotalPrice.sum()})

cltv_df.columns=cltv_df.columns.droplevel(0)
cltv_df.columns=["recency",'T','frequency',"monetary"]
cltv_df["monetary"]=cltv_df["monetary"]/cltv_df["frequency"]
cltv_df.describe().T
cltv_df=cltv_df[(cltv_df["frequency"]> 1)]
cltv_df["recency"]=cltv_df["recency"]/7

#BG-NBD Modelinin Kurulması
bgf=BetaGeoFitter(penalizer_coef=0.01)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

#2010-2011 yıllarındaki veriyi kullanarak İngiltere’deki müşteriler için 6 aylık CLTV tahmini yapınız
cltv_df["expected_purc_6_month"]=bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"])

plot_period_transactions(bgf)
plt.show()

#Gamma-Gamma Modeli
ggf= GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"],cltv_df["monetary"])

cltv_df["expected_average_profit"]=ggf.conditional_expected_average_profit(cltv_df["frequency"],
                            cltv_df["monetary"])

#2010-2011 yıllarındaki veriyi kullanarak İngiltere’deki müşteriler için 6 aylık CLTV tahmini yapınız

cltv_six_months=ggf.customer_lifetime_value(bgf,cltv_df["frequency"],
                                     cltv_df["recency"],
                                     cltv_df["T"],
                                     cltv_df["monetary"],
                                     time=6, #aylik
                                     freq="W",
                                     discount_rate=0.01)
cltv_six_months=cltv_six_months.reset_index()

#: 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
cltv_one_months=ggf.customer_lifetime_value(bgf,cltv_df["frequency"],
                                     cltv_df["recency"],
                                     cltv_df["T"],
                                     cltv_df["monetary"],
                                     time=1, #aylik
                                     freq="W",
                                     discount_rate=0.01)
cltv_one_months=cltv_one_months.reset_index()

cltv_twelve_months=cltv_one_months=ggf.customer_lifetime_value(bgf,cltv_df["frequency"],
                                     cltv_df["recency"],
                                     cltv_df["T"],
                                     cltv_df["monetary"],
                                     time=12, #aylik
                                     freq="W",
                                     discount_rate=0.01)
cltv_twelve_months=cltv_twelve_months.reset_index()

cltv_final = pd.merge(cltv_df, cltv_one_months, on='Customer ID', how='left')
cltv_final = pd.merge(cltv_final, cltv_six_months, on='Customer ID', how='left')
cltv_final = pd.merge(cltv_final, cltv_twelve_months, on='Customer ID', how='left')

#CLTV ye göre Segment Oluşturmak
cltv_final["segment"]=pd.qcut(cltv_final["clv"],4,labels=["D","C","B","A"])
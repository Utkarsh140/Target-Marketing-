# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:37:29 2021

@author: 500060658
"""

from flask import Flask, redirect, url_for, render_template, request, Response
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
sns.set()
import pickle
import csv
import io
import matplotlib
matplotlib.use('Agg')
from collections import Counter

app = Flask(__name__)

products = pickle.load(open('products.pickle', 'rb'))
aisles = pickle.load(open('aisles.pickle', 'rb'))
orders = pickle.load(open('orders.pickle', 'rb'))
prod_order = pickle.load(open('prod_order.pickle', 'rb'))

d1 = pickle.load(open('cluster1.pickle', 'rb'))
d2 = pickle.load(open('cluster2.pickle', 'rb'))
d3 = pickle.load(open('cluster3.pickle', 'rb'))
d4 = pickle.load(open('cluster4.pickle', 'rb'))
log = [d1,d2,d3,d4]

pro_aisle = pickle.load(open('aisle_dictionary.pickle', 'rb'))
cc = pickle.load(open('customer_classification_model.pickle', 'rb'))

@app.route("/", methods = ["POST", "GET"])
def home():
    if request.method == "POST":
        item = request.form["item"]
        global res
        """data_order = []
        data_product = []
        stream1 = io.StringIO(f1.stream.read().decode("UTF8"), newline=None)
        csvfile1 = csv.reader(stream1)
        for row in csvfile1:
            data_order.append(row)
        stream2 = io.StringIO(f2.stream.read().decode("UTF8"), newline=None)
        csvfile2 = csv.reader(stream2)
        for row in csvfile2:
            data_product.append(row)
        order = pd.DataFrame(data_order, columns = ['order_id', 'user_id', 'eval_set', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'])
        prod_order = pd.DataFrame(data_product, columns = ['order_id', 'product_id', 'add_to_cart_order', 'reordered'])
        order.drop(0)
        order.drop(0)
        prod_order.drop(0)
        prod_order.drop(0)
        order = order.astype({"order_id": float, "user_id": int, "order_number": int, "order_dow": int, "order_hour_of_day": int, "days_since_prior_order": float})
        prod_order = prod_order.astype({"order_id": int, "product_id": int, "add_to_cart_order": int, "reordered": int})
        print(prod_order.info())"""
        data = merge_data()
        segmented_data, customer_ids = segment_creation(data)
        customer_segments = cust_seg(segmented_data, customer_ids)
        res = customer_list(item, customer_segments)
        return render_template("after_completion.html", item_name = item)
    else:
        return render_template("home.html")

@app.route("/print_res")
def print_res():
    global day_dictionary
    day_dictionary = get_day_dictionary(res[0:500])
    for key in day_dictionary:
        day_dictionary[key] = day_dictionary[key][0:10]
    return render_template("show_cust.html", result = day_dictionary)


@app.route('/plot1.png')
def plot_png1():
    fig1 = graph1()
    output1 = io.BytesIO()
    FigureCanvas(fig1).print_png(output1)
    return Response(output1.getvalue(), mimetype='image/png')

@app.route('/plot2.png')
def plot_png2():
    fig2 = graph2()
    output2 = io.BytesIO()
    FigureCanvas(fig2).print_png(output2)
    return Response(output2.getvalue(), mimetype='image/png')

@app.route('/plot3.png')
def plot_png3():
    fig3 = graph3()
    output3 = io.BytesIO()
    FigureCanvas(fig3).print_png(output3)
    return Response(output3.getvalue(), mimetype='image/png')

@app.route('/plot4.png')
def plot_png4():
    fig4 = graph4()
    output4 = io.BytesIO()
    FigureCanvas(fig4).print_png(output4)
    return Response(output4.getvalue(), mimetype='image/png')

def merge_data():       
    data = pd.merge(prod_order,products, on = ['product_id','product_id'])
    data = pd.merge(data,orders,on=['order_id','order_id'])
    data = pd.merge(data,aisles,on=['aisle_id','aisle_id'])
    print("inside merge data")
    return data

def segment_creation(data):
    cust_prod = pd.crosstab(data['user_id'], data['aisle'])
    new_cust_prod = cust_prod.reset_index()
    user_id = new_cust_prod['user_id']
    user_id = user_id.to_dict()
    
    pca = PCA(n_components=6)
    pca.fit(cust_prod)
    pca_samples = pca.transform(cust_prod)
    ps = pd.DataFrame(pca_samples)
    new_data = ps.drop([0,2,3,5], axis = 1)
    print("segment createion")
    return new_data, user_id

def cust_seg(seg_data, customer_ids):
    l = {}
    nd = seg_data.to_numpy()
    for i in range(131209):
        l[customer_ids[i]] = cc.predict(nd[i].reshape(1,-1))[0]
    print("cust seg")
    return l

def customer_list(item, temp):
    a = pro_aisle[item]
    m = 200
    no = -1
    for i in log:
        no = no+1
        if i[a]<m:
            m = i[a]
            segno = no
    list_of_customers = []
    for key in temp:
        if temp[key] == segno:
            list_of_customers.append(key)
    print("customer_list")
    return list_of_customers

def graph1():
    fig1 = plt.figure(figsize=(8,6))
    sns.countplot(x=orders.order_dow, color='blue')
    plt.xlabel('', fontsize=16)
    plt.xticks(fontsize=15)
    plt.ylabel('Order Counts', fontsize=16)
    plt.yticks(fontsize=15)
    return fig1

def graph2():
    fig2 = plt.figure(figsize=(12,8))
    sns.countplot(x=orders.order_hour_of_day, color='green')
    plt.xlabel('Hour of Day', fontsize=16)
    plt.xticks(fontsize=15)
    plt.ylabel('Order Counts', fontsize=16)
    plt.yticks(fontsize=15)
    return fig2

def graph3():
    fig3  = plt.figure(figsize=(12,8))
    sns.countplot(x=orders.days_since_prior_order, color= 'orange')
    plt.xlabel('Days Since Last Purchase', fontsize=16)
    plt.xticks(np.arange(31), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,24, 25, 26, 27, 28, 29, 30],  fontsize=15)
    plt.xticks(rotation='vertical')
    plt.ylabel('Order Counts', fontsize=16)
    plt.yticks(fontsize=15)
    return fig3

def graph4():
    frequency_per_number_of_order = prod_order.groupby('order_id')['product_id'].count().value_counts()
    fig4 = plt.figure(figsize=(20,8))
    sns.barplot(x=frequency_per_number_of_order.index, y=frequency_per_number_of_order.values, color='violet')
    plt.title('Amount of Items Per Order', fontsize=16)
    plt.ylabel('Order Counts', fontsize=16)
    plt.xlabel('Number of Items', fontsize=16)
    plt.xticks(rotation='vertical');
    return fig4

def get_day_dictionary(res):
    day_dict = {}
    for i in res:
        testres = orders[orders['user_id'] == i]['order_dow']
        testres = list(testres)
        c = Counter(testres)
        sor = sorted(c, key = c.get)
        day_dict[i] = sor[-1]
    day_dict2 = {'monday':[], 'tuesday':[], 'wednesday':[], 'thursday':[], 'friday':[], 'saturday':[], 'sunday':[]}
    for key in day_dict:
        if day_dict[key] == 0:
            day_dict2['monday'].append(key)
        elif day_dict[key] == 1:
            day_dict2['tuesday'].append(key)
        elif day_dict[key] == 2:
            day_dict2['wednesday'].append(key)
        elif day_dict[key] == 3:
            day_dict2['thursday'].append(key)
        elif day_dict[key] == 4:
            day_dict2['friday'].append(key)
        elif day_dict[key] == 5:
            day_dict2['saturday'].append(key)
        elif day_dict[key] == 6:
            day_dict2['sunday'].append(key)
    return day_dict2
    
if __name__ == "__main__":
    app.run()
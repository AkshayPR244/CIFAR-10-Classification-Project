# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 18:33:04 2018

@author: saket
"""

import csv
import matplotlib.pyplot as plt
#import numpy as np
import re

with open('logbook/470047.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    lcnnadam01 = []
    acnnadam01 = []
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            lcnnadam01.append(float(row[1]))
            acnnadam01.append(float(row[2])/100)
            line_count += 1
print(f'Processed {line_count} lines.')

with open('logbook/633085.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    lresadam01 = []
    aresadam01 = []
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            lresadam01.append(float(row[1]))
            aresadam01.append(float(row[2])/100)
            line_count += 1
print(f'Processed {line_count} lines.')

with open('logbook/084204.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    lattadam01 = []
    aattadam01 = []
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            lattadam01.append(float(row[1]))
            aattadam01.append(float(row[2])/100)
            line_count += 1
print(f'Processed {line_count} lines.')

with open('logbook/732848.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    lcnnsgd01 = []
    acnnsgd01 = []
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            lcnnsgd01.append(float(row[1]))
            acnnsgd01.append(float(row[2])/100)
            line_count += 1
print(f'Processed {line_count} lines.')

with open('logbook/620972.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    lressgd01 = []
    aressgd01 = []
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            lressgd01.append(float(row[1]))
            aressgd01.append(float(row[2])/100)
            line_count += 1
print(f'Processed {line_count} lines.')

with open('logbook/870526.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    lattsgd01 = []
    aattsgd01 = []
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            lattsgd01.append(float(row[1]))
            aattsgd01.append(float(row[2])/100)
            line_count += 1
print(f'Processed {line_count} lines.')

with open('logbook/043791.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    lcnnadam1 = []
    acnnadam1 = []
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            lcnnadam1.append(float(row[1]))
            acnnadam1.append(float(row[2])/100)
            line_count += 1
print(f'Processed {line_count} lines.')

with open('logbook/097902.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    lresadam1 = []
    aresadam1 = []
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            lresadam1.append(float(row[1]))
            aresadam1.append(float(row[2])/100)
            line_count += 1
print(f'Processed {line_count} lines.')

with open('logbook/596012.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    lattadam1 = []
    aattadam1 = []
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            lattadam1.append(float(row[1]))
            aattadam1.append(float(row[2])/100)
            line_count += 1
print(f'Processed {line_count} lines.')

fig = plt.figure()
ax = plt.axes()

x = list(range(len(lcnnsgd01)))
ax.plot(x,lattadam1)
ax.plot(x,aattadam1)
ax.legend(['Loss','Accuracy'])
ax.set(title="Structure3: SelfAttention Adam lrate 0.001", xlabel="batches done", ylabel="loss and accuracy")


import numpy as np
import pandas as pd
import csv
import math


def Find_Maxmin():
    source_file = "kddcup.data_10_percent_corrected_result_2.csv"
    handled_file = "kddcup.data_10_percent_corrected_result_minmax_2.csv"
    dic = {}
    data_file = open(handled_file,'w',newline='')
    with open(source_file,'r') as data_source:
        csv_reader=csv.reader(data_source)
        count = 0
        row_num = ""
        for row in csv_reader:
            count = count+1
            row_num = row
        with open(source_file,'r') as data_source:
            csv_reader=csv.reader(data_source)
            final_list = list(csv_reader)
            print(final_list)
            jmax = []
            jmin = []
            for k in range(0, len(final_list)):
                jmax.append(max(final_list[k]))
                jmin.append(min(final_list[k]))
            jjmax = float(max(jmax))
            jjmin = float(min(jmin))
            listss = []
            for i in range(0,len(row_num)):
                lists = []
                with open(source_file,'r') as data_source:
                    csv_reader=csv.reader(data_source)
                    for row in csv_reader:
                        if (jjmax-jjmin) == 0:
                            x = 0
                        else:
                            x = (float(row[i])-jjmin) / (jjmax-jjmin)
                        lists.append(x)
                listss.append(lists)
            for j in range(0,len(listss)):
                dic[j] = listss[j]
            df = pd.DataFrame(data = dic)
            df.to_csv(data_file,index=False,header=False)
            data_file.close()

if __name__ == '__main__':
    Find_Maxmin()


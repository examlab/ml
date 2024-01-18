import pandas as pd
data=pd.read_csv("datas_z.txt")
print(data)
total_records=len(data)
print("The total number of records are :",total_records)


golf=len(data[data['recreation']=='golf'])
print("Number of recreation as golf :",golf)


prob_gold=golf/total_records
print("Unconditional Probability of golf is :",prob_golf)


medrisk=len(data[data['risk']=='medRisk'])
print("Records having risk as medrisk :",medrisk)


medrisk_single=len(data[data['risk']=='medRisk' & (data['status']=='single')])
print("Records having risk as medrisk and status as single :",medrisk_single)


prob_medrisk_single=medrisk_single/total_records
prob_medrisk=medrisk/total_records
con_prob=prob_medrisk_single/prob_medrisk
print("The probability of status being single given risk is medrisk :",round(con_prob,3))




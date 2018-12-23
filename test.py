from datetime import datetime

print(None == ' ')

# datetime_object1 = datetime.strptime('20100629-03:23', '%Y%m%d-%H:%M')
# datetime_object2 = datetime.strptime('20100729-03:23', '%Y%m%d-%H:%M')
#
# if(datetime_object1 >= datetime_object2):
#     print(True)
# else:
#     print(False)

# sqlContext = SQLContext(sc)

# house_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('DelhiWeather.csv')
# house_df.take(1)
#
# house_df.cache()
# house_df.printSchema()

# data = sc.textFile("DelhiWeather.csv") \
#     .map(lambda line: line.split(",")) \
#     .filter(lambda line: len(line) > 5) \
#
# header = data.first()
#
# data = data.filter(lambda line: line != header) \
#     .map(lambda line: (datetime.strptime(line[0], '%Y%m%d-%H:%M'), line[2], line[6], line[8], line[11], line[1])) \
#     .collect()
#
# print(data)

# axs = pd.scatter_matrix(numeric_data, figsize=(8, 8))
#

# type = type(Any Variable)

# n = len(numeric_data.columns)
# for i in range(n):
#     v = axs[i, 0]
#     v.yaxis.label.set_rotation(0)
#     v.yaxis.label.set_ha('right')
#     v.set_yticks(())
#     h = axs[n-1, i]
#     h.xaxis.label.set_rotation(90)
#     h.set_xticks(())
#
# plt.show()
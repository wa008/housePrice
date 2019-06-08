
# 并列柱状图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#设置字体以便支持中文
import numpy as np

# result_before = [99,97.37,98.76,99.15,99.21,97.31]
# result_after = [99,97.37,98.92,99.23,99.23,97.31]
result_before = [97.37,98.76,99.15,99.21,97.31]
result_after = [97.37,98.92,99.23,99.23,97.31]

x=np.arange(len(result_before))#柱状图在横坐标上的位置

bar_width=0.3#设置柱状图的宽度
# tick_label=['lr', 'svm', 'dt', 'rf', 'xgboost', 'bpnn']
tick_label=['SVR', 'dt', 'rf', 'xgboost', 'bpnn']
plt.xticks(fontsize = 39)
plt.yticks(fontsize = 39)

#绘制并列柱状图
plt.bar(x,result_before,bar_width,color='salmon',label='update_parameter_before')
plt.bar(x+bar_width,result_after,bar_width,color='orchid',label='update_parameter_after')


plt.legend()#显示图例，即label
plt.xticks(x+bar_width/2,tick_label)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
plt.show()
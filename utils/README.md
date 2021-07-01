### 辅助代码

王锦宏 于7月1日上午编写



#### 这是一个什么目录？

这里存放着我们组在创建模型时使用的辅助代码。



#### Daily_DataClean

DailyMail_Stories的原文件为UTF-8编码的200000+个.story文件，其格式请见Sample_Raw_Data.story。通过preprocess.ipynb批量清理、整理story文件，并通过Data clean.ipynb产生符合要求的导入csv。格式详见sample_val.csv



#### Rogue_cal

含有通过对比模型输出与原输出计算三种Rogue值的方法。txt文件夹内是模型的样例输出。



#### Clean_modeloutput

因为模型输出倾向于将固定字词多次重复，使用这一工具代码清洗模型输出。
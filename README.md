# DM_team

# 数据挖掘大作业

# 基于豆瓣影评的电影推荐模型进展报告

- 运行代码前，请使用`conda env create --file environment1.yml`安装所需环境

- 文件夹Scrapy_DOUBAN为使用的数据爬取脚本
- wath.py 包括数据清洗以及数据分类标记的代码，环境为environment1.yml
- pltData.py 包括数据标记后的可视化展示的代码，环境为environment1.yml
- train.py 包括使用lstm模型进行情感分析训练的代码，环境为environment1.yml
- predict.py 包括使用上边训练得到的模型进行预测的代码，环境为environment1.yml
- recommand.py 包括对用户进行电影推荐的代码，环境为environment1.yml
- apriori.py 包括关联规则挖掘的代码

由于训练出的结果模型文件`lstm.h5`较大，需要使用的话请联系以获取数据：v6p6prw2@anonaddy.me，或登录校园网进行下载，网址为：http://10.1.0.92:19080/temp_file/lstm.h5 ，或直接在乐学里边下载code.7z，那个里边是完整的

## 如需重新训练，请联系以获取数据：v6p6prw2@anonaddy.me

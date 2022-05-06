# -*- coding: utf-8 -*-
# @Time    : 2022/4/15 15:20
# @Author  : Zhao Zijun
# @Email   : uchihazzj@outlook.com
# @File    : wath.py
# @Software: PyCharm

import csv
import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# def getCsvFrame(filePath):
#     """
#     :param filePath:
#     :type filePath:
#     :return:
#     :rtype:
#     """
#     # csvFrame = pd.read_csv(filePath, header=None, encoding='ISO-8859-1')
#     csvFrame = pd.read_csv(filePath, encoding='ISO-8859-1')
#     return csvFrame


def test(mid):
    # beforeW_filename = "comments.csv"
    beforeW_filename = "movies.csv"
    # beforeW_filename = "person.csv"
    # beforeW_filename = "ratings.csv"
    # beforeW_filename = "users.csv"
    beforeW_folder = os.path.join(BASE_DIR, "data")
    beforeW_filepath = os.path.join(beforeW_folder, beforeW_filename)

    afterW_filename = beforeW_filename
    afterW_folder = os.path.join(BASE_DIR, "data_clean")
    afterW_filepath = os.path.join(afterW_folder, afterW_filename)

    # csvFrame1 = pd.read_csv(beforeW_filepath, header=None, encoding='ISO-8859-1')
    need_data = []
    data_count = 0
    with open(beforeW_filepath, 'r', encoding="utf_8_sig", errors='ignore') as f:
        # csvreader = csv.reader(f)
        # next(csvreader)
        csv_reader = csv.DictReader(f)
        # header : ['\ufeff"COMMENT_ID"', 'USER_MD5', 'MOVIE_ID', 'CONTENT', 'VOTES', 'COMMENT_TIME', 'RATING']
        for row in csv_reader:
            # print(row)
            if row['MOVIE_ID'] == mid:
                # print(row)
                data_count = data_count + 1
                need_data.append(row)
            # else:
            # data_count = data_count + 1
            # print(row)
        # df = pd.DataFrame(washed_data)
        # df.to_csv(afterW_filepath, encoding="utf_8_sig", index=False)
    # print("comments.csv 已完成数据清洗，删除条目数为 : ", data_count)
    # print(csvFrame2)
    print(need_data)
    print(data_count, " done.")


def washCommentsCsv():
    beforeW_filename = "comments.csv"
    # beforeW_filename = "movies.csv"
    # beforeW_filename = "person.csv"
    # beforeW_filename = "ratings.csv"
    # beforeW_filename = "users.csv"
    beforeW_folder = os.path.join(BASE_DIR, "data")
    beforeW_filepath = os.path.join(beforeW_folder, beforeW_filename)

    afterW_filename = beforeW_filename
    afterW_folder = os.path.join(BASE_DIR, "data_clean")
    afterW_filepath = os.path.join(afterW_folder, afterW_filename)

    # csvFrame1 = pd.read_csv(beforeW_filepath, header=None, encoding='ISO-8859-1')
    washed_data = []
    data_count = 0
    with open(beforeW_filepath, 'r', encoding="utf_8_sig", errors='ignore') as f:
        # csvreader = csv.reader(f)
        # next(csvreader)
        csv_reader = csv.DictReader(f)
        # header : ['\ufeff"COMMENT_ID"', 'USER_MD5', 'MOVIE_ID', 'CONTENT', 'VOTES', 'COMMENT_TIME', 'RATING']
        for row in csv_reader:
            # print(row)
            if row['RATING'] != '':
                washed_data.append(row)
            else:
                data_count = data_count + 1
                # print(row)
        df = pd.DataFrame(washed_data)
        df.to_csv(afterW_filepath, encoding="utf_8_sig", index=False)
    print("comments.csv 已完成数据清洗，删除条目数为 : ", data_count)
    # print(csvFrame2)


def genData():
    """
    生成情感分析用的数据
    """
    beforeW_filename = "comments.csv"
    beforeW_folder = os.path.join(BASE_DIR, "data")
    beforeW_filepath = os.path.join(beforeW_folder, beforeW_filename)

    afterW_filename = beforeW_filename
    afterW_folder = os.path.join(BASE_DIR, "data_clean")
    afterW_filepath = os.path.join(afterW_folder, afterW_filename)
    good_filepath = os.path.join(afterW_folder, "good.csv")
    bad_filepath = os.path.join(afterW_folder, "bad.csv")
    good_data = []
    good_count = 0
    bad_data = []
    bad_count = 0
    # temp = dict()
    with open(afterW_filepath, 'r', encoding="utf_8_sig", errors='ignore') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            # print(row)
            temp = dict()
            if float(row['RATING']) > 2.5:
                temp['CONTENT'] = row['CONTENT']
                good_data.append(temp)
                good_count = good_count + 1
            else:
                temp['CONTENT'] = row['CONTENT']
                bad_data.append(temp)
                bad_count = bad_count + 1
        df_good = pd.DataFrame(good_data)
        df_good.to_csv(good_filepath, encoding="utf_8_sig", index=False)
        df_bad = pd.DataFrame(bad_data)
        df_bad.to_csv(bad_filepath, encoding="utf_8_sig", index=False)
    print("good : ", good_count, "\nbad : ", bad_count)


def washMoviesCsv():
    # beforeW_filename = "comments.csv"
    beforeW_filename = "movies.csv"
    # beforeW_filename = "person.csv"
    # beforeW_filename = "ratings.csv"
    # beforeW_filename = "users.csv"
    beforeW_folder = os.path.join(BASE_DIR, "data")
    beforeW_filepath = os.path.join(beforeW_folder, beforeW_filename)

    afterW_filename = beforeW_filename
    afterW_folder = os.path.join(BASE_DIR, "data_clean")
    afterW_filepath = os.path.join(afterW_folder, afterW_filename)

    # csvFrame1 = pd.read_csv(beforeW_filepath, header=None, encoding='ISO-8859-1')
    washed_data = []
    data_count = 0
    with open(beforeW_filepath, 'r', encoding="utf_8_sig", errors='ignore') as f:
        # csvreader = csv.reader(f)
        # next(csvreader)
        csv_reader = csv.DictReader(f)
        # header : ['\ufeff"COMMENT_ID"', 'USER_MD5', 'MOVIE_ID', 'CONTENT', 'VOTES', 'COMMENT_TIME', 'RATING']
        for row in csv_reader:
            # print(row)
            if row['DOUBAN_VOTES'] != '0.0':
                if row['DOUBAN_SCORE'] != '0.0':
                    washed_data.append(row)
                else:
                    data_count = data_count + 1
            else:
                data_count = data_count + 1
                # print(row)
        df = pd.DataFrame(washed_data)
        df.to_csv(afterW_filepath, encoding="utf_8_sig", index=False)
    print("movies.csv 已完成数据清洗，删除条目数为 : ", data_count)
    # print(csvFrame2)


def washPersonCsv():
    # beforeW_filename = "comments.csv"
    # beforeW_filename = "movies.csv"
    beforeW_filename = "person.csv"
    # beforeW_filename = "ratings.csv"
    # beforeW_filename = "users.csv"
    beforeW_folder = os.path.join(BASE_DIR, "data")
    beforeW_filepath = os.path.join(beforeW_folder, beforeW_filename)

    afterW_filename = beforeW_filename
    afterW_folder = os.path.join(BASE_DIR, "data_clean")
    afterW_filepath = os.path.join(afterW_folder, afterW_filename)

    # csvFrame1 = pd.read_csv(beforeW_filepath, header=None, encoding='ISO-8859-1')
    washed_data = []
    data_count = 0
    with open(beforeW_filepath, 'r', encoding="utf_8_sig", errors='ignore') as f:
        # csvreader = csv.reader(f)
        # next(csvreader)
        csv_reader = csv.DictReader(f)
        # header : ['\ufeff"COMMENT_ID"', 'USER_MD5', 'MOVIE_ID', 'CONTENT', 'VOTES', 'COMMENT_TIME', 'RATING']
        for row in csv_reader:
            # print(row)
            if row['PERSON_ID'] != '':
                washed_data.append(row)
            else:
                data_count = data_count + 1
                # print(row)
        df = pd.DataFrame(washed_data)
        df.to_csv(afterW_filepath, encoding="utf_8_sig", index=False)
    print("person.csv 已完成数据清洗，删除条目数为 : ", data_count)


def washRatingsCsv():
    # beforeW_filename = "comments.csv"
    # beforeW_filename = "movies.csv"
    # beforeW_filename = "person.csv"
    beforeW_filename = "ratings.csv"
    # beforeW_filename = "users.csv"
    beforeW_folder = os.path.join(BASE_DIR, "data")
    beforeW_filepath = os.path.join(beforeW_folder, beforeW_filename)

    afterW_filename = beforeW_filename
    afterW_folder = os.path.join(BASE_DIR, "data_clean")
    afterW_filepath = os.path.join(afterW_folder, afterW_filename)

    # csvFrame1 = pd.read_csv(beforeW_filepath, header=None, encoding='ISO-8859-1')
    washed_data = []
    data_count = 0
    with open(beforeW_filepath, 'r', encoding="utf_8_sig", errors='ignore') as f:
        # csvreader = csv.reader(f)
        # next(csvreader)
        csv_reader = csv.DictReader(f)
        # header : ['\ufeff"COMMENT_ID"', 'USER_MD5', 'MOVIE_ID', 'CONTENT', 'VOTES', 'COMMENT_TIME', 'RATING']
        for row in csv_reader:
            # print(row)
            if row['MOVIE_ID'] != '' or row['RATING'] != '':
                washed_data.append(row)
            else:
                data_count = data_count + 1
                # print(row)
        df = pd.DataFrame(washed_data)
        df.to_csv(afterW_filepath, encoding="utf_8_sig", index=False)
    print("ratings.csv 已完成数据清洗，删除条目数为 : ", data_count)
    # print(csvFrame2)


def washUsersCsv():
    # beforeW_filename = "comments.csv"
    # beforeW_filename = "movies.csv"
    # beforeW_filename = "person.csv"
    # beforeW_filename = "ratings.csv"
    beforeW_filename = "users.csv"
    beforeW_folder = os.path.join(BASE_DIR, "data")
    beforeW_filepath = os.path.join(beforeW_folder, beforeW_filename)

    afterW_filename = beforeW_filename
    afterW_folder = os.path.join(BASE_DIR, "data_clean")
    afterW_filepath = os.path.join(afterW_folder, afterW_filename)

    # csvFrame1 = pd.read_csv(beforeW_filepath, header=None, encoding='ISO-8859-1')
    washed_data = []
    data_count = 0
    with open(beforeW_filepath, 'r', encoding="utf_8_sig", errors='ignore') as f:
        # csvreader = csv.reader(f)
        # next(csvreader)
        csv_reader = csv.DictReader(f)
        # header : ['\ufeff"COMMENT_ID"', 'USER_MD5', 'MOVIE_ID', 'CONTENT', 'VOTES', 'COMMENT_TIME', 'RATING']
        for row in csv_reader:
            # print(row)
            if row['USER_MD5'] != '':
                washed_data.append(row)
            else:
                data_count = data_count + 1
                # print(row)
        df = pd.DataFrame(washed_data)
        df.to_csv(afterW_filepath, encoding="utf_8_sig", index=False)
    print("users.csv 已完成数据清洗，删除条目数为 : ", data_count)
    # print(csvFrame2)


if __name__ == "__main__":
    # washCommentsCsv()  # 删除条目数为 :  261770
    # washMoviesCsv()  # 删除条目数为 :  101938
    # washPersonCsv()  # 删除条目数为 :  0
    # washRatingsCsv()  # 删除条目数为 :  0
    # washUsersCsv()  # 删除条目数为 :  0
    # test()
    genData()  # good :  3319292  bad :  847413

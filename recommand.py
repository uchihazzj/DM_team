import sys
import random
import os,math
from operator import itemgetter
import csv
import codecs


import sys
import random
import math
import os
from operator import itemgetter

random.seed(0)
user_sim_mat = {}
matrix = []  #全局变量
matrix2 = []

class UserBasedCF(object):
    ''' TopN recommendation - User Based Collaborative Filtering '''

    def __init__(self):
        self.trainset = {}  # 训练集
        self.testset = {}  # 测试集
        self.initialset = {}  # 存储要推荐的用户的信息
        self.n_sim_user = 30
        self.n_rec_movie = 10

        self.movie_popular = {}
        self.movie_count = 0  # 总电影数量

        print('Similar user number = %d' % self.n_sim_user, file=sys.stderr)
        print('recommended movie number = %d' %
              self.n_rec_movie, file=sys.stderr)

    @staticmethod
    def loadfile(filename):
        ''' load a file, return a generator. '''
        fp = open(filename, 'r', encoding='UTF-8')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            # if i % 100000 == 0:
            #     print ('loading %s(%s)' % (filename, i), file=sys.stderr)
        fp.close()
        print('load %s success' % filename, file=sys.stderr)

    def initial_dataset(self, filename1):
        initialset_len = 0
        with open(filename1, 'r', encoding="utf_8_sig", errors='ignore') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
            # for lines in self.loadfile(filename1):
                users, movies, ratings = row['USER_MD5'], row['MOVIE_ID'], row['RATING']
                # users, movies, ratings = lines.split(',')
                self.initialset.setdefault(users, {})
                self.initialset[users][movies] = ratings
                initialset_len += 1

    def generate_dataset(self, filename2, pivot=1.0):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0

        with open(filename2, 'r', encoding="utf_8_sig", errors='ignore') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
            # for lines in self.loadfile(filename1):
                user, movie, rating = row['USER_MD5'], row['MOVIE_ID'], row['RATING']
            # for line in self.loadfile(filename2):
                # user, movie, rating, _ = line.split('::')
                # user, movie, rating = line.split(',')
                # split the data by pivot
                if random.random() < pivot:  # pivot=0.7应该表示训练集：测试集=7：3
                    self.trainset.setdefault(user, {})
                    self.trainset[user][movie] = rating  # trainset[user][movie]可以获取用户对电影的评分  都是整数
                    trainset_len += 1
                else:
                    self.testset.setdefault(user, {})
                    self.testset[user][movie] = rating
                    testset_len += 1

            print('split training set and test set succ', file=sys.stderr)
            print('train set = %s' % trainset_len, file=sys.stderr)
            print('test set = %s' % testset_len, file=sys.stderr)

    def calc_user_sim(self):
        movie2users = dict()

        for user, movies in self.trainset.items():
            for movie in movies:
                # inverse table for item-users
                if movie not in movie2users:
                    movie2users[movie] = set()
                movie2users[movie].add(user)  # 看这个电影的用户id
                # print(movie)   #输出的是movieId
                # print(movie2users[movie])   #输出的是{'userId'...}
                # print(movie2users)    #movieId:{'userId','userId'...}

                # count item popularity at the same time
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1
        # print ('build movie-users inverse table succ', file=sys.stderr)

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users)
        print('total movie number = %d' % self.movie_count, file=sys.stderr)

        # count co-rated items between users  计算用户之间共同评分的物品
        usersim_mat = user_sim_mat
        # print ('building user co-rated movies matrix...', file=sys.stderr)

        for movie, users in movie2users.items():  # 通过.items()遍历movie2users这个字典里的所有键、值
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    usersim_mat.setdefault(u, {})
                    usersim_mat[u].setdefault(v, 0)
                    usersim_mat[u][v] += 1 / math.log(1 + len(users))  # usersim_mat二维矩阵应该存的是用户u和用户v之间共同评分的电影数目
        # print ('build user co-rated movies matrix succ', file=sys.stderr)

        # calculate similarity matrix
        # print ('calculating user similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 20000

        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1


    def recommend(self, user):
        ''' Find K similar users and recommend N movies. '''
        matrix.clear()   #每次都要清空
        K = self.n_sim_user  # 这里等于20
        N = self.n_rec_movie  # 这里等于10
        rank = dict()  # 用户对电影的兴趣度
        # print(self.initialset[user])
        watched_movies = self.trainset[user]  # user用户已经看过的电影  只包括训练集里的
        # 这里之后不能是训练集
        # watched_movies = self.initialset[user]
        for similar_user, similarity_factor in sorted(user_sim_mat[user].items(),
                                                      key=itemgetter(1), reverse=True)[
                                               0:K]:  # itemgetter(1)表示对第2个域(相似度)排序   reverse=TRUE表示降序
            for imdbid in self.trainset[similar_user]:  # similar_user是items里面的键，就是所有用户   similarity_factor是值，就是对应的相似度
                if imdbid in watched_movies:
                    continue  # 如果该电影用户已经看过，则跳过
                # predict the user's "interest" for each movie
                rank.setdefault(imdbid, 0)  # 没有值就为0
                rank[imdbid] += similarity_factor   #rank[movie]就是各个电影的相似度
                # 这里是把和各个用户的相似度加起来，而各个用户的相似度只是基于看过的公共电影数目除以这两个用户看过的电影数量积
                #print(rank[movie])
        # return the N best movies
       # rank_ = dict()
        rank_ = sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]  #类型是list不是字典了
        for key,value in rank_:
            matrix.append(key)    #matrix为存储推荐的imdbId号的数组
            #print(key)     #得到了推荐的电影的imdbid号
        print(matrix)
        #return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]
        return matrix



class ItemBasedCF(object):
    ''' TopN recommendation - Item Based Collaborative Filtering '''

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_movie = 20
        self.n_rec_movie = 10

        self.movie_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

        # print('Similar movie number = %d' % self.n_sim_movie, file=sys.stderr)
        # print('Recommended movie number = %d' %
        #       self.n_rec_movie, file=sys.stderr)

    @staticmethod
    def loadfile(filename):
        ''' load a file, return a generator. '''
        # data1=np.loadtxt(filename,delimiter=',',dtype=float)
        fp = open(filename, 'r', encoding='UTF-8')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            # if i % 100000 == 0:
            #     print ('loading %s(%s)' % (filename, i), file=sys.stderr)
        fp.close()
        print('load %s succ' % filename, file=sys.stderr)

    def generate_dataset(self, filename, pivot=1.0):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            user, movie, rating = line.split(',')
            # user, movie, rating = np.loadtxt(filename,delimiter=',')
            rating = float(rating)
            # print(type(rating))

            # split the data by pivot
            if random.random() < pivot:
                self.trainset.setdefault(user, {})

                self.trainset[user][movie] = float(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})

                self.testset[user][movie] = float(rating)
                testset_len += 1

        # print('split training set and test set succ', file=sys.stderr)
        print('train set = %s' % trainset_len, file=sys.stderr)
        # print('test set = %s' % testset_len, file=sys.stderr)

    def calc_movie_sim(self):
        ''' calculate movie similarity matrix '''
        print('counting movies number and popularity...', file=sys.stderr)

        for user, movies in self.trainset.items():
            for movie in movies:
                # count item popularity
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1

        # print('count movies number and popularity succ', file=sys.stderr)

        # save the total number of movies
        self.movie_count = len(self.movie_popular)
        print('total movie number = %d' % self.movie_count, file=sys.stderr)

        # count co-rated users between items
        itemsim_mat = self.movie_sim_mat
        # print('building co-rated users matrix...', file=sys.stderr)

        for user, movies in self.trainset.items():
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    itemsim_mat.setdefault(m1, {})
                    itemsim_mat[m1].setdefault(m2, 0)
                    itemsim_mat[m1][m2] += 1 / math.log(1 + len(movies) * 1.0)

        #print('build co-rated users matrix succ', file=sys.stderr)

        # calculate similarity matrix
        #print('calculating movie similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for m1, related_movies in itemsim_mat.items():
            for m2, count in related_movies.items():
                itemsim_mat[m1][m2] = count / math.sqrt(
                    self.movie_popular[m1] * self.movie_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating movie similarity factor(%d)' %
                          simfactor_count, file=sys.stderr)

        #print('calculate movie similarity matrix(similarity factor) succ',
             # file=sys.stderr)
        #print('Total similarity factor number = %d' %
              #simfactor_count, file=sys.stderr)

    def recommend(self, user):
        ''' Find K similar movies and recommend N movies. '''
        K = self.n_sim_movie
        N = self.n_rec_movie
        matrix2.clear()
        rank = {}
        watched_movies = self.trainset[user]

        for movie, rating in watched_movies.items():
            for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(),
                                                           key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += similarity_factor * rating
        # return the N best movies
        #print(sorted(rank.items(), key=itemgetter(1), reverse=True)[:N])
        rank_ = sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]
        for key,value in rank_:
            matrix2.append(key)    #matrix为存储推荐的imdbId号的数组
            #print(key)     #得到了推荐的电影的imdbid号
        print(matrix2)
        #return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]
        return matrix2


if __name__ == '__main__':
    # ratingfile = os.path.join('ml-1m', 'ratings.dat')
    # ratingfile1 = os.path.join('ml-100k', 'insertusers.csv')
    #ratingfile = 'comments.csv'  # 一共671个用户
    #ratingfile2 = os.path.join('static', 'rrtotaltable.csv')

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    beforeW_filename = "ratings.csv"
    afterW_filename = beforeW_filename
    afterW_folder = os.path.join(BASE_DIR, "data_clean")
    ratingfile = os.path.join(afterW_folder, afterW_filename)

    usercf = UserBasedCF()
    userId = '0ab7e3efacd56983f16503572d2b9915'
    # usercf.initial_dataset(ratingfile1)
    usercf.generate_dataset(ratingfile)
    usercf.calc_user_sim()
    # usercf.evaluate()
    usercf.recommend(userId)  # 给用户5推荐了10部电影  输出的是‘movieId’,兴趣度   109444、110148都是用户2已经看过并且评分为4的电影。
# print(sorted(user_sim_mat['2'].items(),key=itemgetter(1), reverse=True)[0:20])    #输出的是{'useId':{'另一个userId':相似度,'其他userId':'相似度...'}}...
#这里输的userId,是要从另一张储存登录用户的userid# 这里输的userId,是要从另一张储存登录用户的userid
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os, math, warnings, math, pickle
from tqdm import tqdm
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
# from datetime import datetime
# from deepctr.feature_column import SparseFeat, VarLenSparseFeat
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.models import Model


# from deepmatch.models import *
# from deepmatch.utils import sampledsoftmaxloss
warnings.filterwarnings('ignore')


data_path = os.getcwd()
save_path = os.getcwd()

metric_recall = True


# debug模式： 从训练集中划出一部分数据来调试代码
def get_all_click_sample(data_path, sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(os.path.join(data_path,'train_click_log.csv'))
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click


# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path='./data_raw/', offline=True):
    if offline:
        all_click = pd.read_csv(os.path.join(data_path, 'train_click_log.csv'))
    else:
        trn_click = pd.read_csv(os.path.join(data_path, 'train_click_log.csv'))
        tst_click = pd.read_csv(os.path.join(data_path, 'testA_click_log.csv'))

        all_click = pd.concat([trn_click, tst_click], ignore_index=True)

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click


def get_item_info_df(data_path):
    item_info_df = pd.read_csv(os.path.join(data_path, 'articles.csv'))

    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})

    return item_info_df


def get_item_emb_dict(data_path):
    """
    读取并处理文章的内容embedding
    :param data_path: 数据集存放路径
    :return: item_emb_dict, 格式为 {article_id: embedding_vector} 的字典
    """
    # 1. 读取文章的embedding数据文件
    if os.path.exists(os.path.join(data_path, 'item_emb_dict.pkl')):
        return pickle.load(open(os.path.join(data_path, 'item_emb_dict.pkl'), 'rb'))
    item_emb_df = pd.read_csv(os.path.join(data_path, 'articles_emb.csv'))

    # 2. 筛选出列名中包含 'emb' 的列，这些列组成了文章的特征向量
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]

    # 3. 将特征列转换为 numpy 数组
    # np.ascontiguousarray 确保数组在内存中是连续存储的，有助于提高后续计算速度
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])

    # 4. 进行 L2 归一化 (Normalization)
    # axis=1 表示对每一行（每一篇文章的向量）求范数
    # keepdims=True 保持二维结构，以便进行广播除法
    # 目的：归一化后，两个向量的点积就等同于它们的余弦相似度，计算更方便
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    # 5. 构建字典：将 article_id 作为键，对应的归一化后的 embedding 向量作为值
    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))

    # 6. 将字典序列化保存到本地文件 'item_content_emb.pkl'，后续可直接加载使用
    pickle.dump(item_emb_dict, open(os.path.join(save_path, 'item_content_emb.pkl'), 'wb'))

    return item_emb_dict

max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))


# 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp')

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(
        lambda x: make_item_time_pair(x)) \
        .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

    return user_item_time_dict


# 根据时间获取商品被点击的用户序列  {item1: [(user1, time1), (user2, time2)...]...}
# 这里的时间是用户点击当前商品的时间，好像没有直接的关系。
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))

    click_df = click_df.sort_values('click_timestamp')
    item_user_time_df = click_df.groupby('click_article_id')[['user_id', 'click_timestamp']].apply(
        lambda x: make_user_time_pair(x)) \
        .reset_index().rename(columns={0: 'user_time_list'})

    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict


# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)

    # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_hist_df, click_last_df


# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段直接使用
def get_item_info_dict(item_info_df):
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)

    item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id'])) #类型
    item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count'])) #字数
    item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts'])) #创建时间

    return item_type_dict, item_words_dict, item_created_time_dict


def get_user_hist_item_info_dict(all_click):
    # 获取user_id对应的用户历史点击文章类型的集合字典
    user_hist_item_typs = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_typs_dict = dict(zip(user_hist_item_typs['user_id'], user_hist_item_typs['category_id'])) #用户阅读类型

    # 获取user_id对应的用户点击文章的集合
    user_hist_item_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_ids_dict = dict(zip(user_hist_item_ids_dict['user_id'], user_hist_item_ids_dict['click_article_id'])) #用户阅读文章

    # 获取user_id对应的用户历史点击的文章的平均字数字典
    user_hist_item_words = all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_hist_item_words_dict = dict(zip(user_hist_item_words['user_id'], user_hist_item_words['words_count'])) #用户阅读文章平均字数

    # 获取user_id对应的用户最后一次点击的文章的创建时间
    all_click_ = all_click.sort_values('click_timestamp')
    user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(
        lambda x: x.iloc[-1]).reset_index()

    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    user_last_item_created_time['created_at_ts'] = user_last_item_created_time[['created_at_ts']].apply(max_min_scaler)

    user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], \
                                                user_last_item_created_time['created_at_ts']))

    return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict

def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click

# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    """
    计算召回方法的命中率 (Hit Rate) 指标
    :param user_recall_items_dict: 字典, 存储每个用户的召回列表 {user_id: [(item_id, score), ...]}
    :param trn_last_click_df: DataFrame, 用户最后一次点击的真实数据 (Ground Truth)
    :param topk: 整数, 评估的截断位置，决定了循环的最大范围
    """

    # 将用户的最后一次点击转换为 {user_id: last_click_item_id} 的字典，方便快速查找
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))

    # 获取需要评估的用户总数
    user_num = len(user_recall_items_dict)

    # 循环评估 top-10, top-20, ... 直到 top-k (注意：这里的步长是10)
    # 比如 topk=50，则分别打印 top10, top20, top30, top40, top50 的命中情况
    for k in range(10, topk + 1, 10):
        hit_num = 0  # 记录在当前 k 值下，命中目标的用户数量

        # 遍历每个用户及其对应的召回列表
        for user, item_list in user_recall_items_dict.items():
            # 1. 获取该用户真实点击的文章 ID (Label)
            if user not in last_click_item_dict:
                continue

            # 2. 截取算法推荐的前 k 个文章 ID (Prediction)
            # x[0] 是文章ID, x[1] 是分数，这里只取 ID
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]

            # 3. 判断是否命中：真实点击是否包含在推荐列表集合中
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1

        # 计算命中率：命中用户数 / 总用户数
        hit_rate = round(hit_num * 1.0 / user_num, 5)

        # 打印当前 K 值的评估结果
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)


def get_user_activate_degree_dict(all_click_df):
    """
    计算用户的活跃度（点击文章的次数），并进行归一化处理
    :param all_click_df: 包含用户点击日志的 DataFrame
    :return: 字典 {user_id: normalized_activity_score}
    """
    # 1. 统计每个用户的点击量
    # groupby('user_id'): 按用户分组
    # ['click_article_id'].count(): 计算每个用户点击了多少篇文章（活跃度绝对值）
    # reset_index(): 将结果转换回 DataFrame 格式，包含 'user_id' 和 'click_article_id'（此时代表计数）两列
    all_click_df_ = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()

    # 2. 用户活跃度归一化 (Min-Max Scaling)
    # 实例化一个归一化工具，将数据缩放到 [0, 1] 区间
    # 公式: (x - min) / (max - min)
    mm = MinMaxScaler()

    # 对点击次数这一列进行拟合和转换
    # 注意：fit_transform 需要二维数组作为输入，所以用了双中括号 [['click_article_id']]
    # 归一化后的目的是消除量纲影响，让活跃度变成一个 0~1 之间的权重值
    all_click_df_['click_article_id'] = mm.fit_transform(all_click_df_[['click_article_id']])

    # 3. 将结果转换为字典格式，方便后续快速查询
    # key: 用户ID, value: 归一化后的活跃度分数
    user_activate_degree_dict = dict(zip(all_click_df_['user_id'], all_click_df_['click_article_id']))

    return user_activate_degree_dict


def countnum(x):
    t = set()
    for i in x:
        t.add(i)
    return len(t)
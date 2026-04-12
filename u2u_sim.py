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
from share import save_path,  get_item_user_time_dict
def usercf_sim(all_click_df, user_activate_degree_dict):
    """
        用户相似性矩阵计算
        :param all_click_df: 数据表
        :param user_activate_degree_dict: 用户活跃度的字典
        return 用户相似性矩阵

        思路: 基于用户的协同过滤(详细请参考上一期推荐系统基础的组队学习) + 关联规则
    """
    item_user_time_dict = get_item_user_time_dict(all_click_df)

    u2u_sim = {}
    user_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, click_time in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v, click_time in user_time_list:
                u2u_sim[u].setdefault(v, 0)
                if u == v:
                    continue
                # 用户平均活跃度作为活跃度的权重，这里的式子也可以改善
                activate_weight = 100 * 0.5 * (user_activate_degree_dict[u] + user_activate_degree_dict[v])
                u2u_sim[u][v] += activate_weight / math.log(len(user_time_list) + 1)

    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])

    # 将得到的相似性矩阵保存到本地
    pickle.dump(u2u_sim_, open(os.path.join(save_path, 'usercf_u2u_sim.pkl'), 'wb'))

    return u2u_sim_

def usercf_sim_copy(all_click_df, user_activate_degree_dict):
    item_user_time_dict = get_item_user_time_dict(all_click_df)

    u2u_sim = {}
    user_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for user, click_time in user_time_list:
            user_cnt[user] += 1
            u2u_sim.setdefault(user, {})
            for v, click_time_ in user_time_list:
                if user == v: continue
                u2u_sim[user].setdefault(v, 0)
                activate_weight = 100 * 0.5 * (user_activate_degree_dict[user] + user_activate_degree_dict[v])
                u2u_sim[user][v] += activate_weight / math.log(len(user_time_list) + 1)
    u2u_sim_ = u2u_sim.copy()
    for user, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim_[user][v] = wij / math.sqrt(user_cnt[user] * user_cnt[v])

    return u2u_sim_


# 基于用户的召回 u2u2i
def user_based_recommend(user_id, user_item_time_dict, u2u_sim, sim_user_topk, recall_item_num,
                         item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
        :param u2u_sim: 字典，文章相似性矩阵
        :param sim_user_topk: 整数， 选择与当前用户最相似的前k个用户
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param item_created_time_dict: 文章创建时间列表
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵

        return: 召回的文章列表 [(item1, score1), (item2, score2)...]
    """
    # 历史交互
    user_item_time_list = user_item_time_dict[user_id]  # [(item1, time1), (item2, time2)..]
    user_hist_items = set([i for i, t in user_item_time_list])  # 存在一个用户与某篇文章的多次交互， 这里得去重

    items_rank = {}
    for sim_u, wuv in sorted(u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:sim_user_topk]:
        for i, click_time in user_item_time_dict[sim_u]:
            if i in user_hist_items:
                continue
            items_rank.setdefault(i, 0)

            loc_weight = 1.0
            content_weight = 1.0
            created_time_weight = 1.0

            # 当前文章与该用户看的历史文章进行一个权重交互
            for loc, (j, click_time) in enumerate(user_item_time_list):
                # 点击时的相对位置权重
                loc_weight += 0.9 ** (len(user_item_time_list) - loc)
                # 内容相似性权重
                if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                    content_weight += emb_i2i_sim[i][j]
                if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                    content_weight += emb_i2i_sim[j][i]

                # 创建时间差权重
                created_time_weight += np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))

            items_rank[i] += loc_weight * content_weight * created_time_weight * wuv

    # 热度补全
    if len(items_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in items_rank.items():  # 填充的item应该不在原来的列表中
                continue
            items_rank[item] = - i - 100  # 随便给个复数就行
            if len(items_rank) == recall_item_num:
                break

    items_rank = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return items_rank

def user_based_recommend_copy(user_id, user_item_time_dict, u2u_sim, sim_user_topk, recall_item_num,
                         item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
        :param u2u_sim: 字典，文章相似性矩阵
        :param sim_user_topk: 整数， 选择与当前用户最相似的前k个用户
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param item_created_time_dict: 文章创建时间列表
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵

        return: 召回的文章列表 [(item1, score1), (item2, score2)...]
    """

    user_item_hist = user_item_time_dict[user_id]
    user_hist_items = set(i for i, t in user_item_time_dict[user_id])
    item_rank = {}
    for sim_u, wuv in sorted(u2u_sim[user_id].items(), key=lambda x: x[1], reverse= True)[:sim_user_topk]:
        for item_i, click_time in user_item_time_dict[sim_u]:
            if item_i in user_hist_items: continue
            item_rank.setdefault(item_i, 0)

            loc_weight = 1.0  # 位置权重，衡量位置关系
            content_weight = 1.0  # 内容权重，衡量内容相似度
            created_time_weight = 1.0  # 创建时间权重，衡量文章创建时间差
            for loc, (item_j, click_time) in enumerate(user_item_hist):

                loc_weight += 0.9 ** (len(user_item_hist) - loc)
                if emb_i2i_sim.get(item_i, {}).get(item_j, None) is not None:
                    content_weight += emb_i2i_sim[item_i][item_j]
                if emb_i2i_sim.get(item_j, {}).get(item_i, None) is not None:
                    content_weight += emb_i2i_sim[item_j][item_i]
                created_time_weight += np.exp(0.8 ** np.abs(item_created_time_dict[item_i] - item_created_time_dict[item_j]))

            item_rank[item_i] += loc_weight * content_weight * created_time_weight * wuv
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank: continue
            item_rank.setdefault(item, 0)
            item_rank[item] = - i - 100
            if len(item_rank) == recall_item_num: break

    return item_rank


# 使用Embedding的方式获取u2u的相似性矩阵，用youtubednn的用户embeding向量来计算用户之间的相似度，得到u2u的相似性矩阵
# topk指的是每个user, faiss搜索后返回最相似的topk个user
def u2u_embdding_sim(click_df, user_emb_dict, save_path, topk):
    user_list = []
    user_emb_list = []
    for user_id, user_emb in user_emb_dict.items():
        user_list.append(user_id)
        user_emb_list.append(user_emb)

    user_index_2_rawid_dict = {k: v for k, v in zip(range(len(user_list)), user_list)}

    user_emb_np = np.array(user_emb_list, dtype=np.float32)

    # 建立faiss索引
    user_index = faiss.IndexFlatIP(user_emb_np.shape[1])
    user_index.add(user_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = user_index.search(user_emb_np, topk)  # 返回的是列表

    # 将向量检索的结果保存成原始id的对应关系
    user_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(user_emb_np)), sim, idx)):
        target_raw_id = user_index_2_rawid_dict[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = user_index_2_rawid_dict[rele_idx]
            user_sim_dict[target_raw_id][rele_raw_id] = user_sim_dict.get(target_raw_id, {}).get(rele_raw_id,
                                                                                                 0) + sim_value

    # 保存i2i相似度矩阵
    pickle.dump(user_sim_dict, open(os.path.join(save_path, 'youtube_u2u_sim.pkl'), 'wb'))
    return user_sim_dict

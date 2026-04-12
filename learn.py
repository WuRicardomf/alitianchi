from share import *

# 采样数据
all_click_df = get_all_click_sample(data_path)

# 全量训练集
# all_click_df = get_all_click_df(data_path, offline=False)

# 对时间戳进行归一化,用于在关联规则的时候计算权重
all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)

item_info_df = get_item_info_df(data_path)

item_emb_dict = get_item_emb_dict(data_path)

# 获取文章的属性信息，保存成字典的形式方便查询
item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)

# 定义一个多路召回的字典，将各路召回的结果都保存在这个字典当中
user_multi_recall_dict =  {'itemcf_sim_itemcf_recall': {},
                           'embedding_sim_item_recall': {},
                           'youtubednn_recall': {},
                           'youtubednn_usercf_recall': {},
                           'cold_start_recall': {}}

trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)

from i2i_sim import itemcf_sim
i2i_sim = itemcf_sim(all_click_df, item_created_time_dict)

# 由于usercf计算时候太耗费内存了，这里就不直接运行了
# 如果是采样的话，是可以运行的
from u2u_sim import usercf_sim
user_activate_degree_dict = get_user_activate_degree_dict(all_click_df)
u2u_sim = usercf_sim(all_click_df, user_activate_degree_dict)


# 向量检索相似度计算
# topk指的是每个item, faiss搜索后返回最相似的topk个item
def embdding_sim(click_df, item_emb_df, save_path, topk):
    """
        基于内容的文章embedding相似性矩阵计算
        :param click_df: 数据表
        :param item_emb_df: 文章的embedding
        :param save_path: 保存路径
        :patam topk: 找最相似的topk篇
        return 文章相似性矩阵

        思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
    """

    # 文章索引与文章id的字典映射
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))

    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    # 向量进行单位化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    # 建立faiss索引
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_index.add(item_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = item_index.search(item_emb_np, topk)  # 返回的是列表

    # 将向量检索的结果保存成原始id的对应关系
    item_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id,
                                                                                                 0) + sim_value

    # 保存i2i相似度矩阵
    pickle.dump(item_sim_dict, open(os.path.join(save_path,'emb_i2i_sim.pkl'), 'wb'))

    return item_sim_dict

embdding_sim(all_click_df, pd.read_csv(os.path.join(save_path, 'articles_emb.csv')),  save_path, topk=10)
# 获取双塔召回时的训练验证数据
# negsample指的是通过滑窗构建样本的时候，负样本的数量


from typing import List, Sequence, Optional


# python

import torch.nn as nn
from torch_rechub.models.matching import YoutubeDNN
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gen_model_input, generate_seq_feature_match
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model_weights(model, filepath='model_weights.pth'):
    """
    保存模型参数
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    # 保存模型参数
    torch.save(model.state_dict(), filepath)
    print(f"模型参数已保存到: {filepath}")

    # 可选：保存模型结构（如果需要完整的模型）
    # torch.save(model, 'full_model.pth')


# 4. 加载模型参数的函数
def load_model_weights(model, filepath='model_weights.pth'):
    """
    加载模型参数
    """
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
        model.eval()  # 设置为评估模式
        print(f"模型参数已从 {filepath} 加载")
        return model
    else:
        print(f"文件 {filepath} 不存在")
        return None


def youtubednn_u2i_dict(data, topk=20):
    sparse_features = ["click_article_id", "user_id"]
    SEQ_LEN = 30  # 用户点击序列的长度，短的填充，长的截断

    user_profile_ = data[["user_id"]].drop_duplicates('user_id')
    item_profile_ = data[["click_article_id"]].drop_duplicates('click_article_id')

    # 类别编码
    features = ["click_article_id", "user_id", "click_environment", "click_deviceGroup", "click_os", "click_country", "click_region", "click_referrer_type"]
    feature_max_idx = {}

    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])
        feature_max_idx[feature] = data[feature].max() + 1

    # 提取user和item的画像，这里具体选择哪些特征还需要进一步的分析和考虑
    user_profile = data[['user_id', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type']].drop_duplicates('user_id')
    item_profile = data[["click_article_id"]].drop_duplicates('click_article_id')

    user_index_2_rawid = dict(zip(user_profile['user_id'], user_profile_['user_id']))
    item_index_2_rawid = dict(zip(item_profile['click_article_id'], item_profile_['click_article_id']))

    # print(user_index_2_rawid)

    user_col, item_col = 'user_id', 'click_article_id'

    df_train, df_test = generate_seq_feature_match(
        data, user_col, item_col,
        time_col="click_timestamp",
        item_attribute_cols=[],
        sample_method=1,
        mode=2,  # list-wise 模式
        neg_ratio=3,
        min_item=0
    )
    user_cols = ['user_id', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type']
    x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=SEQ_LEN)
    # list-wise 训练：label 固定为 0 表示第一个位置是正样本
    # print(x_train)
    y_train = np.array([0] * df_train.shape[0])
    x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=SEQ_LEN)
    # 确定Embedding的维度
    embedding_dim = 16

    # 将数据整理成模型可以直接输入的形式
    user_feature = [SparseFeature(name, vocab_size=feature_max_idx[name],embed_dim=embedding_dim) for name in user_cols]
    user_feature += [SequenceFeature('hist_click_article_id', vocab_size=feature_max_idx['click_article_id'], embed_dim=embedding_dim, pooling="mean", shared_with='click_article_id')]
    item_feature = [SparseFeature('click_article_id', vocab_size=feature_max_idx['click_article_id'], embed_dim=embedding_dim)]
    neg_item_feature = [SequenceFeature('neg_items', vocab_size=feature_max_idx['click_article_id'], embed_dim=embedding_dim, pooling='concat', shared_with='click_article_id')]

    # print(user_feature)
    model = YoutubeDNN(user_features=user_feature,
    item_features=item_feature,
    neg_item_feature=neg_item_feature,
    user_params={"dims": [128, 64, embedding_dim]},
    temperature=1.0).to(device)

    from torch_rechub.trainers import MatchTrainer
    save_dir = './saved/youtube_dnn/'
    torch.manual_seed(2022)

    trainer = MatchTrainer(
        model,
        mode=2,  # list-wise 训练模式
        optimizer_params={
            "lr": 1e-4,
            "weight_decay": 1e-6
        },
        n_epoch=100,
        device=device,
        model_path=save_dir
    )
    all_item = df_to_dict(item_profile)
    test_user = x_test

    # 创建 DataLoader

    import torch.nn.functional as F
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    dg = MatchDataGenerator(x=x_train, y=y_train)
    train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=2048)
    n_epoch = 100
    trainer.fit(train_dl)
    torch.save(model.state_dict(), save_dir + 'model.pth')
    user_embedding = trainer.inference_embedding(
        model=model, mode="user",
        data_loader=test_dl,
        model_path="./saved/youtube_dnn/"
    )
    item_embedding = trainer.inference_embedding(
        model=model, mode="item",
        data_loader=item_dl,
        model_path="./saved/youtube_dnn/"
    )
    load_model_weights(model, filepath=save_dir + 'model.pth')
    # for epoch in range(1, n_epoch + 1):
    #     model.train()
    #     total_loss = 0
    #     for step, (train_x, train_y) in enumerate(train_dl):
    #         optimizer.zero_grad()
    #         for k in train_x:
    #             train_x[k] = train_x[k].to(device)
    #         train_y = train_y.to(device)
    #         pred = model(train_x)
    #         loss = F.cross_entropy(pred, train_y)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #         if step % 10 == 0:
    #             print(f"epoch {epoch} step {step}, loss={loss.item():.4f}")
    #
    #     avg_loss = total_loss / step
    #     if total_loss < 1e-3:
    #         break
    #     print(f"epoch {epoch} finished, avg_loss={avg_loss:.4f}")
    # 训练完成后，保存模型参数
    # save_model_weights(model, filepath='./saved/youtube_dnn/model.pth')
#     from torch_rechub.utils.match import  Annoy
# #user向量
#     model.mode = 'user'
#     user_embedding = torch.empty((0, embedding_dim), device=device)
#     item_embedding = torch.empty((0, embedding_dim), device=device)
#     model.eval()
#     test_user = torch.tensor([]).to(device)
#     for user in test_dl:
#         for x in user:
#             user[x] = user[x].to(device)
#             if x == 'user_id':
#                 test_user = torch.cat((test_user, user[x]),dim=0)
#         # print(user_embedding.shape, model(user).shape)
#         user_embedding = torch.cat((user_embedding, model(user)), dim=0)
#     numpy_test_user = test_user.clone().cpu().detach().numpy()
#     # train_user = torch.tensor([]).to(device)
#     # for user, _ in train_dl:
#     #     print(type(user))
#     #     for x in user:
#     #         print(x)
#     #         user[x] = user[x].to(device)
#     #         if x == 'user_id':
#     #             train_user = torch.cat((train_user, user[x]),dim=0)
#     # numpy_train_user = train_user.clone().cpu().detach().numpy()
#     # print(countnum(numpy_test_user), countnum(numpy_train_user))
#     # breakpoint()
#     # item向量
#     model.mode = 'item'
#     for item in item_dl:
#         for x in item:
#             item[x] = item[x].to(device)
#         item_embedding = torch.cat((item_embedding, model(item)), dim=0)
#
#     # user_embedding_ = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path=save_dir)
#     # item_embedding_ = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=save_dir)
#     # print(user_embedding ==  user_embedding_)
#     # print(item_embedding == item_embedding_)
#     trainer.fit(train_dl)
#     # for train_x, y in train_dl:
#     #     for k in train_x:
#     #         train_x[k] = train_x[k].to(device)
#     #     y = y.to(device)
#     #     pred = model(train_x)
#     #     loss = F.cross_entropy(pred, y)
#     #     print(loss.item())
    numpy_user_embedding = user_embedding.cpu().detach().numpy()
    numpy_item_embedding = item_embedding.cpu().detach().numpy()

    # numpy_user_embedding = numpy_user_embedding / np.linalg.norm(numpy_user_embedding, axis=1, keepdims=True)
    # numpy_item_embedding = numpy_item_embedding / np.linalg.norm(numpy_item_embedding, axis=1, keepdims=True)



    raw_user_id_emb_dict = {user_index_2_rawid[idx]: emb for idx, emb in zip(user_profile['user_id'].values, numpy_user_embedding)}
    raw_item_id_emb_dict = {item_index_2_rawid[idx]: emb for idx, emb in zip(item_profile['click_article_id'].values, numpy_item_embedding)}


    pickle.dump(raw_user_id_emb_dict, open(os.path.join(save_path, 'user_youtube_emb.pkl'), 'wb'))
    pickle.dump(raw_item_id_emb_dict, open(os.path.join(save_path, 'item_youtube_emb.pkl'), 'wb'))



    index = faiss.IndexFlatIP(embedding_dim)
    # 归一化
    faiss.normalize_L2(numpy_user_embedding)
    faiss.normalize_L2(numpy_item_embedding)
    index.add(numpy_item_embedding)  # 将item向量构建索引
    sim, idx = index.search(np.ascontiguousarray(numpy_user_embedding), topk)  # 通过user去查询最相似的topk个item

    user_recall_items_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(x_test['user_id'], sim, idx)):
        target_raw_id = user_index_2_rawid[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list, sim_value_list):
            rele_raw_id = item_index_2_rawid[rele_idx]
            user_recall_items_dict[target_raw_id][rele_raw_id] = user_recall_items_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value

    user_recall_items_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in
                              user_recall_items_dict.items()}
    # 将召回的结果进行排序

    # 保存召回的结果
    # 这里是直接通过向量的方式得到了召回结果，相比于上面的召回方法，上面的只是得到了i2i及u2u的相似性矩阵，还需要进行协同过滤召回才能得到召回结果
    # 可以直接对这个召回结果进行评估，为了方便可以统一写一个评估函数对所有的召回结果进行评估
    pickle.dump(user_recall_items_dict, open(os.path.join(save_path,'youtube_u2i_dict.pkl'), 'wb'))
    return user_recall_items_dict

# 由于这里需要做召回评估，所以讲训练集中的最后一次点击都提取了出来
if not metric_recall:
    user_multi_recall_dict['youtubednn_recall'] = youtubednn_u2i_dict(all_click_df, topk=20)
else:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    user_multi_recall_dict['youtubednn_recall'] = youtubednn_u2i_dict(trn_hist_click_df, topk=20)
    # 召回效果评估
    print("youtubednn recall:")
    metrics_recall(user_multi_recall_dict['youtubednn_recall'], trn_last_click_df, topk=20)

breakpoint()

from i2i_sim import item_based_recommend
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)

i2i_sim = pickle.load(open(os.path.join(save_path,'itemcf_i2i_sim.pkl'), 'rb'))
emb_i2i_sim = pickle.load(open(os.path.join(save_path, 'emb_i2i_sim.pkl'), 'rb'))

sim_item_topk = 20
recall_item_num = 10
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict,
                                                        i2i_sim, sim_item_topk, recall_item_num,
                                                        item_topk_click, item_created_time_dict, emb_i2i_sim)

user_multi_recall_dict['itemcf_sim_itemcf_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['itemcf_sim_itemcf_recall'], open(os.path.join(save_path, 'itemcf_recall_dict.pkl'), 'wb'))

if metric_recall:
    # 召回效果评估
    print("itemcf_sim and embdding_sim with itemcf recall: ")
    metrics_recall(user_multi_recall_dict['itemcf_sim_itemcf_recall'], trn_last_click_df, topk=recall_item_num)

# 这里是为了召回评估，所以提取最后一次点击
# 由于usercf中计算user之间的相似度的过程太费内存了，全量数据这里就没有跑，跑了一个采样之后的数据
from u2u_sim import user_based_recommend
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)

u2u_sim = pickle.load(open(os.path.join(save_path, 'usercf_u2u_sim.pkl'), 'rb'))

sim_user_topk = 20
recall_item_num = 10
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk,
                                                        recall_item_num, item_topk_click, item_created_time_dict,
                                                        emb_i2i_sim)

pickle.dump(user_recall_items_dict, open(os.path.join(save_path, 'usercf_u2u2i_recall.pkl'), 'wb'))

if metric_recall:
    # 召回效果评估
    print("usercf_sim and embdding_sim with usercf recall: ")
    metrics_recall(user_recall_items_dict, trn_last_click_df, topk=recall_item_num)


from u2u_sim import u2u_embdding_sim
user_emb_dict = pickle.load(open(os.path.join(save_path, 'user_youtube_emb.pkl'),  'rb'))
u2u_sim = u2u_embdding_sim(all_click_df, user_emb_dict, save_path, topk=10)

# 使用召回评估函数验证当前召回方式的效果
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
u2u_sim = pickle.load(open(os.path.join(save_path, 'youtube_u2u_sim.pkl'), 'rb'))

sim_user_topk = 20
recall_item_num = 10

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk,
                                                        recall_item_num, item_topk_click, item_created_time_dict,
                                                        emb_i2i_sim)

user_multi_recall_dict['youtubednn_usercf_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['youtubednn_usercf_recall'],open(os.path.join(save_path, 'youtubednn_usercf_recall.pkl'), 'wb'))

if metric_recall:
    # 召回效果评估
    print("youtubednn and embdding_sim with usercf recall: ")
    metrics_recall(user_multi_recall_dict['youtubednn_usercf_recall'], trn_last_click_df, topk=recall_item_num)


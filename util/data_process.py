import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import datetime as dt
import time
import calendar as cd
import arrow
from sklearn import preprocessing
from numpy import newaxis
import math
import random
import itertools
import re
import multiprocessing as mp
from functools import partial
import copy
from util.tools import *


class DataProcessor(object):
    '''
        数据处理器
    '''

    def __init__(self, config):
        self.config = config
        self.data_cfg = config['data']
        self.pre_cfg = config['preprocess']
        self.train_cfg = config['training']
        self.after_cfg = config['after_training']

    @info
    def get_roads_graph(self, ):
        '''
            获取路网图
        input:
            config
        output:
            nx.graph or nx.digraph
        '''
        # 对于新增加的100397交叉口，不在路网图中
        roadnet = pd.read_csv(self.data_cfg['road_detail'], usecols=[
                              'SECTIONID', 'UPROADID', 'DOWNROADID'])
        crossroad_name = pd.read_csv(self.data_cfg['roadname'])
        # 根据路网建立图
        if self.pre_cfg['direction_graph']:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        roadnet_1 = roadnet.apply(lambda x: (
            x['UPROADID'], x['DOWNROADID'], {"id": x['SECTIONID']}), axis=1)
        G.add_edges_from(roadnet_1)

        roadnet_file = pd.read_csv(self.data_cfg['roadnet'])
        roadnet_2 = roadnet_file.apply(
            lambda x: (x['uproadID'], x['downroadID']), axis=1)
        G.add_edges_from(roadnet_2)

        # 添加交叉口属性：name, embedding
        # for i in range(len(crossroad_name)):
        #     G.nodes[crossroad_name.iloc[i]['crossroadID']]['name'] = crossroad_name.iloc[i]['crossroadName']
        print("[Preprocessor] Complete get road graph.")
        return G

    @info
    def get_graph_adjacency(self, G):
        '''
        获取图中的邻接矩阵的字典形式
        '''
        adj_list = nx.to_dict_of_lists(G)

        return adj_list

    @info
    def get_crossroads_embedding(self,):
        '''
            对全部交叉口编码
        input:
            config, 
        output:
            embedding (id list, code list)
            返回的crossroad id是来自于文件，而不是来自于路网图
        '''
        # 对道路使用one hot编码，但是源文件中，一个编码可能对应多个路口交叉关系
        oh_encoder = preprocessing.OneHotEncoder(sparse=False)
        crossroad_file = pd.read_csv(self.data_cfg['roadname'])
        crossroads_id = crossroad_file['crossroadID'].unique()
        roads_pair = crossroad_file['crossroadName'].apply(
            lambda x: str(x)[:-3].split('与')).to_list()
        roads = np.array(roads_pair).reshape([-1, ])
        roads_unique = np.sort(pd.Series(roads).unique())
        roads_unique = roads_unique.reshape(-1, 1)
        roads_embedding = oh_encoder.fit_transform(roads_unique)

        # 对每个交叉口计算出其交叉的全部道路列表，由于数量不一致，所以需要使用字典
        id_dict = dict([(x, []) for x in crossroads_id])
        id_group = crossroad_file.groupby(crossroad_file['crossroadID'])
        for k in id_dict.keys():
            for i in id_group.indices[k]:
                try:
                    id_dict[k].append(roads_pair[i])
                except Exception as e:
                    print(e)
            id_dict[k] = np.unique(np.array(id_dict[k]).reshape((-1)))

        crossroad_code = []

        assert len(id_dict.keys()) == len(crossroads_id)
        for i in crossroads_id:
            crossroad_code.append(oh_encoder.transform(
                np.array(id_dict[i]).reshape((-1, 1))).sum(axis=0).tolist())

        assert len(crossroads_id) == len(crossroad_code)

        return (crossroads_id, crossroad_code)

    @info
    def get_datetime_embedding(self,):
        '''
        获取时间编码
            周期：
            年of年、季度of年、月of年、日of周、日of月、时of日、分of时、秒of分
        '''
        date_index_start = self.pre_cfg['dateindex_start']
        date_index_end = self.pre_cfg['dateindex_end']
        time_index_start = self.pre_cfg['timeindex_start']
        time_index_end = self.pre_cfg['timeindex_end']
        time_index_start = dt.datetime.strptime(
            time_index_start, "%H:%M:%S").time()
        time_index_end = dt.datetime.strptime(
            time_index_end, "%H:%M:%S").time()
        span = self.pre_cfg['time_span']
        datetime_range = pd.date_range(
            start=date_index_start, end=date_index_end, freq=str(span)+'Min')
        datetime_stamps = datetime_range[[x.time() >= time_index_start and x.time(
        ) <= time_index_end for x in datetime_range]]

        T = [1, 4, 12, 7, 0, 24, 60, 60]
        PI = math.pi

        if isinstance(datetime_stamps, dt.datetime):
            timestamps_list = [arrow.get(datetime_stamps)]
        else:
            timestamps_list = [arrow.get(x) for x in datetime_stamps]

        embedding_list = []
        for ts in timestamps_list:
            T[4] = cd.monthrange(ts.year, ts.month)[1]
            d_of_w = cd.weekday(ts.year, ts.month, ts.day)
            q_of_y = math.ceil(ts.month/3)
            datetime_vec = [ts.year, q_of_y, ts.month,
                            d_of_w, ts.day, ts.hour, ts.minute, ts.second]
            x = np.divide(np.array(datetime_vec), np.array(T))
            sin_ = [math.sin(2*PI*i) for i in x]
            cos_ = [math.cos(2*PI*i) for i in x]
            embedding = sin_ + cos_
            embedding_list.append(embedding)
        # 转换为字符串
        datetime_stamps = [x.strftime('%Y-%m-%d %H:%M:%S')
                           for x in datetime_stamps]
        return datetime_stamps, embedding_list

    @info
    def test_diff_from_graph_and_file(self, graph_node_list, file_node_list):
        # 交集
        intersection = [i for i in graph_node_list if i in file_node_list]
        # 并集
        union = list(set(graph_node_list).union(set(file_node_list)))
        # 差集
        diff_g = set(graph_node_list).difference(set(file_node_list))
        diff_f = set(file_node_list).difference(set(graph_node_list))

        print("The difference number between intersection and union is %d" %
              (len(union)-len(intersection)))
        print("The diffrent length of graph nodes to file nodes is %d" %
              (len(diff_g)))
        print("The diffrent length of file nodes to graph nodes is %d" %
              (len(diff_f)))

        # 返回图中节点相对于文件中节点的差集
        return list(diff_g)

    @info
    def cal_traffic_flow(self,):
        '''
            计算交通流
        input:
            short=True
        output: 
            traffic_flow_train, traffic_flow_test
        '''

        span = str(self.pre_cfg['time_span']) + "min"

        traffic_flow_train = pd.DataFrame()
        traffic_flow_test = pd.DataFrame()

        # 计算训练集车流量
        for idx in self.data_cfg['training_files_index']:
            file_name = self.data_cfg['training_files'] + str(idx) + '.csv'
            try:
                data = pd.read_csv(file_name, usecols=[
                    'timestamp', 'crossroadID', 'vehicleID'])
            except Exception as e:
                print(e)
                continue
            datetime_index = pd.to_datetime(data['timestamp'])
            data = data.set_index(datetime_index, drop=True).sort_index()
            # 对车流量数据进行聚合运算
            print("[INFO] Preprocessing training data file %s in %d lines..." %
                  (file_name, data.shape[0]))
            resampled_data = data.groupby('crossroadID')[
                'vehicleID'].resample(span).count()
            # 对结果进行拼接
            traffic_flow_train = pd.concat(
                [traffic_flow_train, resampled_data], axis=0)

        multi_index = pd.MultiIndex.from_tuples(
            traffic_flow_train.index, names=['crossroadID', 'timestamp'])
        traffic_flow_train = traffic_flow_train.set_index(
            multi_index, drop=True).rename(columns={0: 'count'})

        # 计算测试集车流量，仅保留7:00-7:30,9:00-9:30,11:00-11:30,14:00-14:30,16:00-16:30,18:00-18:30
        for idx in self.data_cfg['testing_files_index']:
            file_name = self.data_cfg['testing_files'] + str(idx) + '.csv'
            data = pd.read_csv(file_name, usecols=[
                               'timestamp', 'crossroadID', 'vehicleID'])
            datetime_index = pd.to_datetime(data['timestamp'])
            data = data.set_index(datetime_index, drop=True).sort_index()
            # 对车流量数据进行聚合运算
            print("[INFO] Preprocessing testing data file %s in %d lines..." %
                  (file_name, data.shape[0]))
            resampled_data = data.groupby('crossroadID')[
                'vehicleID'].resample(span).count()
            # 对结果进行拼接
            traffic_flow_test = pd.concat(
                [traffic_flow_test, resampled_data], axis=0)

        multi_index = pd.MultiIndex.from_tuples(
            traffic_flow_test.index, names=['crossroadID', 'timestamp'])
        traffic_flow_test = traffic_flow_test.set_index(
            multi_index, drop=True).rename(columns={0: 'count'})

        return traffic_flow_train, traffic_flow_test

    @info
    def save_traffic_flow(self, traffic_flow_train=None, traffic_flow_test=None,):
        '''
            保存交通流量数据
        input: 
            traffic_flow_train 
            traffic_flow_test 
            short=True
        output:
             None
        '''

        now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        data_dir = self.data_cfg['data_dir']
        save_fname = os.path.join(data_dir,
                                  self.data_cfg['save_path'])
        file_path = save_fname + "_train_" + now + ".csv"
        traffic_flow_train.to_csv(file_path, encoding="utf-8-sig", mode="a")
        print("Training data file saved to %s" % file_path)
        file_path = save_fname + "_test_" + now + ".csv"
        traffic_flow_test.to_csv(file_path, encoding="utf-8-sig", mode="a")
        print("Training data file saved to %s" % file_path)

    @info
    def load_traffic_flow(self,):
        '''
        从目录下获取已经处理好的车流量数据，自动选取最新文件
        '''
        train_file_list = []
        test_file_list = []
        train_newest = time.localtime(0)
        test_newest = time.localtime(0)
        data_dir = self.data_cfg['data_dir']
        # 遍历根目录下所有文件名，查找出含有train和test的文件，提取文件的最后修改时间属性
        for f in os.listdir(data_dir):
            f_path = os.path.join(data_dir, f)
            if f_path.find(self.data_cfg['save_path']+"_train") > -1:
                mtime = time.localtime(os.stat(f_path).st_mtime)
                train_file_list.append((f_path, mtime))
                if mtime > train_newest:
                    train_newest = mtime
            elif f_path.find(self.data_cfg['save_path']+"_test") > -1:
                mtime = time.localtime(os.stat(f_path).st_mtime)
                test_file_list.append((f_path, mtime))
                if mtime > test_newest:
                    test_newest = mtime
        # 将最新的文件找到并读入
        for (f, m) in train_file_list:
            if m == train_newest:
                traffic_flow_train = pd.read_csv(
                    f, index_col=self.data_cfg['traffic_data_columns'])
                print('Success in loading training file %s' % f)
        for (f, m) in test_file_list:
            if m == test_newest:
                traffic_flow_test = pd.read_csv(
                    f, index_col=self.data_cfg['traffic_data_columns'])
                print('Success in loading testing file %s' % f)

        return traffic_flow_train, traffic_flow_test

    @info
    def get_train_test_data(self,
                            traffic_data=None,
                            crossroad_id=None,
                            crossroad_em_dict=None,
                            datetime_em_dict=None,
                            ):
        '''
            获取训练测试数据，按照训练窗口大小切割数据，并以mask分割训练集和测试集，卡口ID和时间戳作为二重索引
        input:
            traffic_data, list, 总数据表，含二重索引和标签，包含数据集合测试集
            crossroad_id, 来自路网图的完整节点列表
            crossroad_em,路网图节点编码，字典(id, embedding)
            datetime_em,时间编码，字典(id, embedding)
        output:
            包含训练集合测试集的所有训练数据
            [time_embedding + crossroad_embedding, traffic_flow] shape:(16+c_em ,6)
        '''
        if traffic_data is None:
            raise ValueError("traffic data must be valid.")
        assert isinstance(traffic_data, list)
        try:
            all_data = pd.concat([traffic_data[0], traffic_data[1]], axis=0)
        except Exception as e:
            print(e)
            print("[ERROR] Please input data list. ")

        tag_length = self.pre_cfg['flow_length']
        time_span = self.pre_cfg['time_span']
        c_id = list(crossroad_em_dict.keys())

        # 首先处理每日的时间索引 这个是固定的可以并行的 然后通过1-19天索引并行处理每天的数据
        daily_start = dt.datetime.strptime(
            self.pre_cfg['timeindex_start'], '%H:%M:%S')
        daily_end = dt.datetime.strptime(
            self.pre_cfg['timeindex_end'], '%H:%M:%S')
        daily_range = pd.date_range(
            start=daily_start, end=daily_end, freq=str(time_span)+'Min')

        daily_indice = pd.Series(daily_range).apply(
            lambda x: pd.date_range(start=x,
                                    end=x+dt.timedelta(
                                        minutes=time_span*tag_length,
                                        seconds=-1),
                                    freq=str(time_span)+'Min'))

        daily_time_index = daily_indice.apply(
            lambda x: [dt.datetime.strftime(x_i, '%H:%M:%S') for x_i in x])[:-tag_length+1]

        days = self.data_cfg['training_files_index'] + \
            self.data_cfg['testing_files_index']

        # 定义在数据字典上的锁
        with mp.Manager() as mgr:
            dict_lock = mgr.Lock()
            # 以crossid为键的字典结构是传入的共享变量，这个字典是子进程与主进程共享的么？

            c_id_dict = mgr.dict()
            c_id_dict['c_id'] = dict([(x, np.array([])) for x in crossroad_id])
            job = []

            # 测试子进程
            # test = self.get_daily_train_test_data(1, daily_time_index, all_data, c_id_dict, dict_lock, crossroad_id)

            for day in days[:]:
                p = mp.Process(target=self.get_daily_train_test_data, args=(
                               day, daily_time_index, all_data, c_id_dict, dict_lock, crossroad_id))
                p.start()
                job.append(p)

            for j in job:
                j.join()

            final_dict = copy.deepcopy(c_id_dict['c_id'])

        return final_dict

    def get_daily_train_test_data(self,
                                  day=None,
                                  daily_time_index=None,
                                  traffic_data=None,
                                  c_id_dict=None,
                                  lock=None,
                                  crossroad_id=None):
        '''
        用于多进程的获取每日的切片数据
        '''
        # 计数器，分别记录处理成功的和失败的key与timestamps
        key_count = 0
        err_key_count = 0
        ts_count = 0
        err_ts_count = 0
        # 计时器
        timer_start = time.time()
        print('[INFO] Day %d : Data is processing...' % day)
        time.sleep(1)
        # 得到当日的所有数据 只在这个数据切片中组合出训练数据
        date_start = dt.datetime.strptime(
            self.pre_cfg['dateindex_start'], '%Y-%m-%d')
        curr_date = str(date_start.year) + '-' + \
            str(date_start.month) + '-' + str(day)
        curr_date = dt.datetime.strptime(curr_date, '%Y-%m-%d')

        curr_traffic_data = traffic_data[pd.DatetimeIndex(
            traffic_data.index.get_level_values(1)).floor('D') == curr_date]
        print('[MSG] Day %d : Data is at length of %d.' %
              (day, len(curr_traffic_data)))

        # 拼接成完整的当日的时间戳
        d_t_index = np.array([])
        for group_index in daily_time_index:
            group = np.array([])
            for t_index in group_index:
                dt_index = curr_date.strftime('%Y-%m-%d') + ' ' + t_index
                group = np.hstack([group, dt_index])
            d_t_index = np.concatenate([d_t_index, group], axis=0)
        d_t_index = d_t_index.reshape((-1, len(group)))
        # 暂存本次处理的字典
        tmp_dict = dict([(x, np.array([])) for x in crossroad_id])

        for c_id_key in crossroad_id:
            # 对指定节点的数据进行处理
            try:
                try_c_id = curr_traffic_data.loc[(c_id_key,)]
            except Exception as e:
                # print('[ERR] c_id_key error',e)
                err_key_count += 1
                continue
            key_count += 1
            sliced_data = np.array([])
            for d_t_i in d_t_index:
                # 对指定时间段进行处理
                sub_list = np.array([])
                for i in d_t_i:
                    # 提取每个时间点的数据，并组合成一个[id, timestamp, value]的数组
                    try:
                        _data = curr_traffic_data.loc[(c_id_key, i)]
                    except Exception as e:
                        # 出现缺失数据，填入nan
                        # print('[ERR] with id %s , datetimestamp error %s' %(str(c_id_key), e))
                        err_ts_count += 1
                        sub_list = np.hstack([sub_list, [c_id_key, i, np.nan]])
                    else:
                        ts_count += 1
                        sub_list = np.hstack(
                            [sub_list, [c_id_key, i, _data.values[0]]])

                # 将每个slice的数据纵向堆叠
                sub_list = sub_list.reshape((-1, 3))
                # 在数据处理时就进行平滑计算
                sub_list = self._fill_missing_avg(sub_list)

                if sliced_data.size == 0:
                    sliced_data = copy.deepcopy(sub_list)
                else:
                    sliced_data = np.vstack([sliced_data, sub_list])

            assert len(sub_list) == self.pre_cfg['flow_length']
            sliced_data = sliced_data.reshape((-1, len(sub_list), 3))
            tmp_dict[c_id_key] = copy.deepcopy(sliced_data)

        lock.acquire()
        remote_dict = c_id_dict['c_id']
        for k in remote_dict.keys():
            if remote_dict[k].size == 0:
                tmp = copy.deepcopy(tmp_dict[k])
                remote_dict[k] = tmp
            else:
                tmp1 = np.array(copy.deepcopy(tmp_dict[k]))[:]
                # print(tmp1)
                tmp2 = np.array(copy.deepcopy(remote_dict[k]))[:]
                # print(tmp2)
                try:
                    tmp3 = np.concatenate([tmp2, tmp1], axis=0)
                except Exception as e:
                    print('[EXCP --]')
                    print(e)
                    print(k, tmp1.shape, tmp2.shape)

                    print('[-- EXCP]')
                    tmp3 = tmp2
                remote_dict[k] = tmp3
                # print('Success concatenating Day %d data.' %day)
        c_id_dict['c_id'] = remote_dict
        lock.release()

        timer_end = time.time()
        print('[INFO] Day %d : Data is processed in %d sec.' %
              (day, timer_end - timer_start))
        print("[MSG] Day %d : Total keys is %d , total timstamps is %d " %
              (day, key_count+err_key_count, ts_count+err_ts_count))
        print("[MSG] Day %d : Success finding %d keys and %d timestamps data , and fail in %d keys and %d timestamps data."
              % (day, key_count, ts_count, err_key_count, err_ts_count))

    def _fill_missing_avg(self, sub_list):
        '''
        对缺失数据进行平滑，只要是序列有两个有效数据，就使用线性回归，
        一个以下有效数据，直接舍弃
        '''
        from sklearn.linear_model import LinearRegression

        seq = sub_list[:, 2]
        seq = [float(x) for x in seq]

        if np.isnan(seq).sum() < int(len(seq)-1) and np.isnan(seq).sum() > 0:
            x = range(len(seq))
            y = seq

            x_t = np.array([i for i in x if np.isnan(seq[i]) == False])
            y_t = np.array([i for i in y if np.isnan(i) == False])

            lr = LinearRegression()
            lr.fit(x_t.reshape(-1, 1), y_t)

            x_p = np.array([i for i in x if np.isnan(seq[i])])
            y_p = lr.predict(x_p.reshape(-1, 1))

            for i in range(len(x_p)):
                seq[x_p[i]] = max(y_p[i], 0.0)
            sub_list[:, 2] = list(seq)
        elif np.isnan(seq).sum() >= int(len(seq)-1):
            sub_list[:, 2] = np.nan

        return sub_list

    @info
    def save_train_test_data(self, training_data):
        '''
        将已经切片好的训练测试数据保存到本地文件
        '''
        import pickle

        data_dir = self.data_cfg['data_dir']
        save_fname = os.path.join(data_dir,
                                  'data.pkl')

        with open(save_fname, 'wb') as out:
            pickle.dump(training_data, out)

        print("Data is saved as \' %s \' ." % save_fname)

    @info
    def load_train_test_data(self,):
        '''
        将切片数据加载到本地
        '''
        import pickle

        data_dir = self.data_cfg['data_dir']
        save_fname = os.path.join(data_dir,
                                  'data.pkl')

        with open(save_fname, 'rb') as inputdata:
            read_dict = pickle.load(inputdata)

        print('Data is loaded from \'data.pkl \' .')
        return read_dict

    @info
    def get_dataset_mask(self, mask_num=35, seed_num=None, crossroad_id=None):
        '''
        以随机数获取数据集上的mask
        '''
        if seed_num is None:
            seed_num = 1337
        if crossroad_id is None:
            raise ValueError("Crossroad_id should be valid list.")
        assert mask_num > 0 and mask_num < len(crossroad_id)
        crossroad_id = list(crossroad_id)

        random.seed(seed_num)
        masked_id = random.sample(crossroad_id, mask_num)

        print("[MASK] Set mask at %s " % (time.strftime(
            '%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

        return masked_id

    @info
    def encode_train_test_data(self, crossroad_em=None, datetime_em=None, train_test_data=None, norm=True):
        '''
        对训练测试数据进行嵌入编码，并对车流量进行标准化
        crossroad_em:(id, embedding)
        datetime_em:(dt, embedding)
        train_test_data:[:, timesteps, dim]
        norm:Normalisation
        missing_avg:缺失数据平滑
        '''
        c_id_em_dict = dict(zip(crossroad_em[0], crossroad_em[1]))
        dt_em_dict = dict(zip(datetime_em[0], datetime_em[1]))

        train_test_embedding = dict()

        for k in train_test_data.keys():
            if train_test_data[k].size == 0:
                continue
            if k == int(100406):
                # 目前只有100406这个节点在训练数据中没有编码，先跳过
                continue
            c_data = train_test_data[k]
            # 进行嵌入编码，转换为float列表
            embeddings = np.apply_along_axis(
                self._traffic_flow_embedding, 2, c_data, c_id_em_dict, dt_em_dict, norm)

            train_test_embedding[k] = embeddings

        return train_test_embedding

    def _traffic_flow_embedding(self, raw_data, c_id_em_dict=None, dt_em_dict=None, norm=True):
        '''
        对车流量数据进行标准化，被embed函数调用的嵌入时间戳和节点的函数
        raw_data:[id, timestamp, count]
        需要对count中的none数据进行处理
        '''
        flow_max = self.data_cfg['flow_max']
        c_em = c_id_em_dict[int(raw_data[0])]
        dt_em = dt_em_dict[raw_data[1]]
        value = float(raw_data[2])

        if norm:
            if value != np.nan:
                norm_flow = float(value/flow_max)
        else:
            norm_flow = value

        return np.concatenate([c_em, dt_em, [norm_flow]])

    @info
    def serialize_data(self, data_dict, shuffle=False, batch_size=32):
        '''
        将数据字典合并成一个长的张量
        '''
        s_data = np.array([])
        for k in data_dict:
            assert data_dict[k].shape[1:] == (6, 411)

            if s_data.size == 0:
                s_data = copy.deepcopy(data_dict[k])
            else:
                s_data = np.concatenate([s_data, data_dict[k]], axis=0)

        if shuffle:
            s_data = np.random.shuffle(s_data)

        return s_data

    def data_generator(self, data_dict, batch_size=32):
        '''
        生成数据，尽量避免拼接数据字典的值，减少内存占用
        生成数据为 ([batch_size, 6, 410], [batch_size, 6, 1])
        '''
        g_data = np.array([])
        batch_data = np.array([])

        key_list = [k for k, v in data_dict.items()]
        np.random.shuffle(key_list)
        while 1:
            for k in key_list:
                assert data_dict[k].shape[1:] == (6, 411)
                g_data = copy.deepcopy(data_dict[k])
                np.random.shuffle(g_data)
                for i in range(g_data.shape[0]):
                    if np.isnan(g_data[i][:, -1]).sum() > 0:
                        continue
                    if batch_data.shape[0] == 0:
                        batch_data = g_data[i]
                        batch_data = batch_data[newaxis, :, :, ]
                    elif batch_data.shape[0] < batch_size:
                        batch_data = np.concatenate(
                            [batch_data, g_data[i][newaxis, :, :, ]], axis=0)
                    elif batch_data.shape[0] == batch_size:
                        batch_data = batch_data.reshape((-1, 6, 411))
                        x = batch_data[:, :, :-1]
                        y = batch_data[:, :, -1]
                        yield (x, y)
                        batch_data = np.array([])

    @info
    def get_predict_data(self, crossroad_em=None, datetime_em=None):
        '''
        获取待预测的数据编码
        '''
        c_id_em_dict = dict(zip(crossroad_em[0], crossroad_em[1]))
        dt_em_dict = dict(zip(datetime_em[0], datetime_em[1]))
        timesteps = self.pre_cfg['flow_length']

        submit_filename = self.data_cfg['submit_data']
        submit_col = self.data_cfg['submit_data_col']
        submit_example = pd.read_csv(submit_filename)
        submit_date = submit_example[submit_col[0]].unique()
        submit_ids = submit_example[submit_col[1]].unique()
        submit_time = submit_example[submit_col[2]].unique()
        submit_time = ['0'+x if len(x) < 5 else x for x in submit_time]

        date_start = self.pre_cfg['dateindex_start']
        date_end = self.pre_cfg['dateindex_end']
        date_range = pd.date_range(start=date_start, end=date_end, freq='D')
        submit_date = [x.date() for x in date_range if x.day in submit_date]
        datetime_range = [dt.datetime.strftime(
            x, '%Y-%m-%d')+' '+y+':00' for x in submit_date for y in submit_time]

        predict_embedding = dict()
        dt_em = [dt_em_dict[x] for x in datetime_range]
        for k in submit_ids:
            try:
                id_em = c_id_em_dict[k]
            except Exception as e:
                # 100397这个交叉口不在字典里
                # 12月5日更新后又加进来了。。。
                print(e, )
            pred_em = [np.concatenate([id_em, dt_em_i]) for dt_em_i in dt_em]
            pred_em = np.array(pred_em).reshape(
                int(len(dt_em)/timesteps), timesteps, -1)
            predict_embedding[k] = pred_em

        return predict_embedding

    @info
    def load_predict_data(self,):
        '''
        从文件中载入最新的结果
        '''
        import pickle

        file_list = []
        newest = time.localtime(0)
        data_dir = self.after_cfg['save_result_path']
        data_file = 'results_'
        # 遍历根目录下所有文件名，查找出含有train和test的文件，提取文件的最后修改时间属性
        for f in os.listdir(data_dir):
            f_path = os.path.join(data_dir, f)
            if f_path.find(data_file) > -1:
                mtime = time.localtime(os.stat(f_path).st_mtime)
                file_list.append((f_path, mtime))
                if mtime > newest:
                    newest = mtime
        # 将最新的文件找到并读入
        for (f, m) in file_list:
            if m == newest:
                with open(f, 'rb') as inputdata:
                    result_data = pickle.load(inputdata)
                print('Success in loading result file %s' % f)

        return result_data

    @info
    def output_predict_csv(self, predict_data=None, result=None, datetime_em=(None, None)):
        '''
        将预测结果处理成csv文件
        '''
        dt_em_dict = dict(zip(datetime_em[0], datetime_em[1]))
        dt_em_len = len(datetime_em[1][0])

        submit_filename = self.data_cfg['submit_data']
        submit_col = self.data_cfg['submit_data_col']
        submit_example = pd.read_csv(submit_filename)
        submit_date = submit_example[submit_col[0]].unique()
        submit_ids = submit_example[submit_col[1]].unique()
        submit_time = submit_example[submit_col[2]].unique()
        submit_time = ['0'+x if len(x) < 5 else x for x in submit_time]

        date_start = self.pre_cfg['dateindex_start']
        date_end = self.pre_cfg['dateindex_end']
        date_range = pd.date_range(start=date_start, end=date_end, freq='D')
        submit_date = [x.date() for x in date_range if x.day in submit_date]
        datetime_range = [dt.datetime.strftime(
            x, '%Y-%m-%d')+' '+y+':00' for x in submit_date for y in submit_time]

        sub_dt_em_dict = {k: v for k,
                          v in dt_em_dict.items() if k in datetime_range}
        output = pd.DataFrame(columns=submit_col)

        for k in predict_data.keys():
            assert k in result.keys()
            pred_x = predict_data[k]
            pred_dt_x = pred_x[:, :, -dt_em_len:]
            res = result[k]
            # 预测数据时间编码数组 [24, 6, 16] 结果数据数组[24, 6, 1]
            assert pred_dt_x.shape[:1] == res.shape[:1]
            pred_dt_x_reshape = pred_dt_x.reshape((-1, pred_dt_x.shape[-1]))
            res_reshape = res.reshape((-1, res.shape[-1]))
            for x in range(len(pred_dt_x_reshape)):
                # 利用字典子集查找 提高查找效率
                dt_x = [k for k, v in sub_dt_em_dict.items() if (
                    v == pred_dt_x_reshape[x]).all()]
                assert len(dt_x) == 1
                dt_x_timestamp = dt.datetime.strptime(
                    dt_x[0], '%Y-%m-%d %H:%M:%S')
                timebegin = str(dt_x_timestamp.hour) + ':' + \
                    str(dt_x_timestamp.minute)
                if timebegin[0] == '0':
                    timebegin = timebegin[1:]

                ouput_line = {
                    submit_col[0]: str(dt_x_timestamp.day),
                    submit_col[1]: str(k),
                    submit_col[2]: timebegin,
                    submit_col[3]: res_reshape[x][0]
                }

                output = pd.concat([output, pd.DataFrame(
                    ouput_line, index=[0])], ignore_index=True)
        output = pd.DataFrame.set_index(output, submit_col[:2], drop=True)
        output = output.sort_index()
        print(output)
        output_path = self.after_cfg['output_path']
        if not os.path.exists(self.after_cfg['output_path']):
            os.makedirs(self.after_cfg['output_path'])
        now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_path,
                                   self.after_cfg['output_file']+now+'.csv')
        output.to_csv(output_file, encoding='utf-8')

        print('[RESULT] Submit file is saved in %s' % output_file)


class DataVisualization(object):
    '''
        数据可视化
    attr：
        config
    method:
        draw_graph
        describe_graph
    '''

    def __init__(self, config):
        self.visual_cfg = config['visualization']

    def draw_graph(self, G):
        # 绘制图的节点和边
        plt.figure()
        node_num = G.number_of_nodes()
        edge_num = G.number_of_edges()
        edge_colors = range(edge_num)
        node_colors = range(node_num)

        nx.draw(G, pos=nx.spring_layout(G),
                with_labels=True,
                node_size=10,
                node_color=node_colors,
                cmap=plt.cm.Reds,
                edge_color=edge_colors,
                edge_cmap=plt.cm.Blues,
                width=0.5,
                font_size=6
                )

        plt.show()

    def describe_graph(self, G):
        # 打印图的基本信息
        print(nx.info(G))

        print("[INFO] The nodes of the Graph.")
        print(nx.nodes(G))

        print("[INFO] The adjacency of the Graph.")
        print(nx.to_pandas_adjacency(G))

        print("[INFO] The edgelist of the Graph.")
        print(nx.to_pandas_edgelist(G))

        print("[INFO] The edge info of the Graph.")
        print(G.edges.data())

    def describe_traffic_data(self, traffic_data=None):
        '''
        对交通流量统计数据进行describe
        '''
        print(traffic_data.describe())
        null_data = traffic_data.isnull()
        na_data = traffic_data.isna()
        print("The sum of missing data is %d, the sum of Nan data is %d." %
              (null_data.sum(), na_data.sum()))

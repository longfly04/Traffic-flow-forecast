import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import datetime as dt
from sklearn import preprocessing
from numpy import newaxis
import math
import random

class DataPreprocesseor(object):
    '''
        数据集预处理器
    attr:
        config
    method:
        get_roads_graph：获取路网图
        get_roads_embedding：获取道路编码
        get_crossroads_code：获取交叉口编码

        save_traffic_flow：保存中间数据，车流量
        cal_traffic_flow：计算车流量（长短期）
        get_train_test_data：分割训练集测试集
        get_dataset_mask：获取数据集的mask
    '''

    def __init__(self, config):
        self.config = config
        self.data_cfg = config['data']
        self.pre_cfg = config['preprocess']

    def get_roads_graph(self, ):
        '''
            获取路网图
        input:
            config
        output:
            nx.graph or nx.digraph
        '''
        roadnet = pd.read_csv(self.data_cfg['road_detail'], usecols=[
                              'SECTIONID', 'UPROADID', 'DOWNROADID'])
        crossroad_name = pd.read_csv(self.data_cfg['roadname'])
        # 根据路网建立图
        if self.pre_cfg['direction_graph']:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        roadnet_tuple = roadnet.apply(lambda x: (
            x['UPROADID'], x['DOWNROADID'], {"id": x['SECTIONID']}), axis=1)
        G.add_edges_from(roadnet_tuple)
        # 添加交叉口属性：name, embedding

        for i in range(len(crossroad_name)):
            G.nodes[crossroad_name.iloc[i]['crossroadID']
                    ]['name'] = crossroad_name.iloc[i]['crossroadName']

        return G

    def get_crossroads_code(self,):
        '''
            对全部交叉口编码
        input:
            config, 
        output:
            embedding (id list, code list)
        '''
        # 对道路使用one hot编码
        oh_encoder = preprocessing.OneHotEncoder(sparse=False)
        crossroad_name = pd.read_csv(self.data_cfg['roadname'])
        crossroads = crossroad_name['crossroadName'].apply(
            lambda x: str(x)[:-3].split('与')).to_list()
        roads = np.array(crossroads).reshape([-1, ])
        roads = np.sort(pd.Series(roads).unique())
        roads = roads.reshape(-1, 1)
        roads_embedding = oh_encoder.fit_transform(roads)
        # print(crossroads)
        # 对全部交叉口编码
        crossroad_code = []

        for i in range(len(crossroads)):
            crossroad_code.append(oh_encoder.transform(
                np.array(crossroads[i]).reshape([-1, 1])).sum(axis=0))
            
        assert len(crossroad_name) == len(crossroad_code)
        
        return (crossroad_name['crossroadID'].to_list(), crossroad_code)

    def cal_traffic_flow(self, short=True):
        '''
            计算交通流量（长短期）
        input:
            short=True
        output: 
            traffic_flow_train, traffic_flow_pred
        '''
        if short:
            span = str(self.pre_cfg['short_span']) + "min"
        else:
            span = str(self.pre_cfg['long_span']) + "min"
        traffic_flow_train = pd.DataFrame()
        traffic_flow_pred = pd.DataFrame()

        # 计算训练集车流量
        for idx in self.data_cfg['training_files_index']:
            file_name = self.data_cfg['training_files'] + str(idx) + '.csv'
            data = pd.read_csv(file_name, usecols=[
                               'timestamp', 'crossroadID', 'vehicleID'])
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
        print(traffic_flow_train)

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
            traffic_flow_pred = pd.concat(
                [traffic_flow_pred, resampled_data], axis=0)

        multi_index = pd.MultiIndex.from_tuples(
            traffic_flow_pred.index, names=['crossroadID', 'timestamp'])
        traffic_flow_pred = traffic_flow_pred.set_index(
            multi_index, drop=True).rename(columns={0: 'count'})
        print(traffic_flow_pred)

        return traffic_flow_train, traffic_flow_pred

    def save_traffic_flow(self, traffic_flow_train=None, traffic_flow_pred=None, short=True):
        '''
            保存交通流量数据
        input: 
            traffic_flow_train 
            traffic_flow_pred 
            short=True
        output:
             None
        '''

        if short:
            msg = 'short_term_'
        else:
            msg = 'long_term_'
        now = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        file_path = self.data_cfg['save_path'] + "_train_" + msg + now + ".csv"
        traffic_flow_train.to_csv(file_path, encoding="utf-8-sig", mode="a")
        print("Training data file saved to %s" % file_path)
        file_path = self.data_cfg['save_path'] + "_test_" + msg + now + ".csv"
        traffic_flow_pred.to_csv(file_path, encoding="utf-8-sig", mode="a")
        print("Training data file saved to %s" % file_path)

    def get_train_test_data(self, 
                            traffic_flow_train_short, 
                            traffic_flow_train_long, 
                            crossroad_id, 
                            crossroad_code,
                            ):
        '''
            获取训练测试数据
        input:
            traffic_flow_train_short, 
            traffic_flow_train_long, 
            crossroad_id, 
            crossroad_code,
        output:
            ?
        '''
        short_length = self.pre_cfg['short_flow_length']
        long_length = self.pre_cfg['long_flow_length']

    def get_dataset_mask(self, mask_num=35, seed=None, G=None):
        '''
        以随机数获取数据集上的mask
        '''
        if seed==None:
            seed = 1337
        if G==None:
            raise ValueError("Graph should be valid data.")
        assert mask_num > 0 and mask_num < len(G.nodes())

        nodes_list = list(G.nodes())
        node_choice = random


    def get_datetime_embedding(self, timestamps):
        '''
        获取时间编码
        '''
        datetime_list = pd.datetime(timestamps).to_list()
        sin_em = datetime_list.apply(lambda x: math.sin(x))
        cos_em = datetime_list.apply(lambda x: math.cos(x))



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

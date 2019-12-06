from util.data_process import *
import json
import sys
sys.path.append(
    json.load(open('config.json', 'r', encoding='utf-8'))['f_path'])

def main():
    config = json.load(open('config.json', 'r', encoding='utf-8'))

    data_pro = DataProcessor(config=config)
    data_vis = DataVisualization(config=config)

    # 获取路网图
    G = data_pro.get_roads_graph()

    # 获取节点列表
    adj_list = data_pro.get_graph_adjacency(G)
    keys = list(adj_list.keys())

    # 图的基本信息
    # data_vis.describe_graph(G)

    # 绘制图
    # draw_graph(G)

    # 计算交通流量
    # traffic_flow_train, traffic_flow_test = data_pro.cal_traffic_flow()

    # 将车流量数据保存本地
    # data_pro.save_traffic_flow(traffic_flow_train, traffic_flow_test)

    # 测试时间编码
    # now = dt.datetime.now()
    # t_em = data_pro.get_datetime_embedding(now)
    # print(t_em)

    # 载入已经计算好的车流量数据文件
    traffic_flow_train, traffic_flow_test = data_pro.load_traffic_flow()

    # 从交叉口文件中获取交叉口id和交叉口编码
    crossroad_id, crossroad_em = data_pro.get_crossroads_embedding()

    # 测试节点的差集
    # 注意，crossroadName.csv文件中交叉口数量相比于roadnet.csv中通过路网路计算出来的交叉口数量少81个
    # 但是不能确定，训练集中是否所有交叉口都在crossroadName.csv中
    diff = data_pro.test_diff_from_graph_and_file(graph_node_list=keys, file_node_list=crossroad_id)

    diff_crossroad = [i for i in keys if i not in crossroad_id]
    diff_crossroad_nei = [adj_list[i] for i in diff_crossroad]
    # sub_adj_list = dict(zip(diff_crossroad, diff_crossroad_nei))
    edge_count = sum([len(i) for i in diff_crossroad_nei])
    # sub_G = nx.parse_adjlist(sub_adj_list)
    # edges = [(v,u,d) for (v,u,d) in G.edges_iter(diff_crossroad_nei) if G.has_edge(v,u)]

    # 时间编码
    datetime_stamps, datetime_index_embedding = data_pro.get_datetime_embedding()

    data_dict = data_pro.get_train_test_data(
        traffic_data=[traffic_flow_train, traffic_flow_test],
        crossroad_id=keys,
        crossroad_em_dict=dict((zip(crossroad_id, crossroad_em))),
        datetime_em_dict=dict(
            (zip(datetime_stamps, datetime_index_embedding))),
    )
    # 保存数据字典，作为每次训练的初始数据{id: array [:, timesteps, dim]}
    data_pro.save_train_test_data(data_dict)




if __name__ == "__main__":
    main()

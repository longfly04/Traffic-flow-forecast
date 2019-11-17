from util.data_preprocess import *
import json
import sys
sys.path.append(json.load(open('config.json', 'r', encoding='utf-8'))['f_path'])
from model.layers import *


def main():
    config = json.load(open('config.json', 'r', encoding='utf-8'))
    data_pre = DataPreprocesseor(config=config)
    data_vis = DataVisualization(config=config)

    # 获取路网图和adj
    G = data_pre.get_roads_graph()
    A = nx.to_pandas_adjacency(G)
    V = G.nodes()

    if config['visualization']['draw_graph']:
        data_vis.draw_graph(G)
    if config['visualization']['describe_graph']:
        data_vis.describe_graph(G)

    # 计算交通微观流量
    traffic_flow_train_s, traffic_flow_pred_s = data_pre.cal_traffic_flow(short=True)
    # 计算交通宏观流量
    traffic_flow_train_l, traffic_flow_pred_l = data_pre.cal_traffic_flow(short=False)

    








if __name__=="__main__":
    main()
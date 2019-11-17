from util.data_preprocess import *
import json
import sys
sys.path.append(json.load(open('config.json', 'r', encoding='utf-8'))['f_path'])
from model.layers import *


def main():
	config = json.load(open('config.json', 'r', encoding='utf-8'))

	data_pre = DataPreprocesseor(config=config)

	# 获取路网图
	G = data_pre.get_roads_graph()
	
	# 图的基本信息
	# describe_graph(G)

	# 绘制图
	# draw_graph(G)

	# 计算交通流量
	traffic_flow_train, traffic_flow_test = data_pre.cal_traffic_flow(config)	

	# 将车流量数据保存本地
	# save_traffic_flow(config, traffic_flow_train, traffic_flow_test)

	Gat = GraphAttention()
	Gat.build()


	pass




if __name__=="__main__":
	main()
from util.data_process import *
import json
import sys
sys.path.append(
    json.load(open('config.json', 'r', encoding='utf-8'))['f_path'])

from model.baseline import *


def main():
    config = json.load(open('config.json', 'r', encoding='utf-8'))
    # 初始化预训练和可视化对象
    data_pro = DataProcessor(config=config)
    # 获取路网图
    G = data_pro.get_roads_graph()
    # 获取节点列表
    adj_list = data_pro.get_graph_adjacency(G)
    keys = list(adj_list.keys())
    # 时间编码
    datetime_stamps, datetime_index_em = data_pro.get_datetime_embedding()
    # 从交叉口文件中获取交叉口id和交叉口编码
    crossroad_id, crossroad_em = data_pro.get_crossroads_embedding()
    # 载入已经计算好的车流量数据文件
    # traffic_flow_train, traffic_flow_test = data_pro.load_traffic_flow()
    # 载入已经处理好的数据字典
    data_dict = data_pro.load_train_test_data()

    predict_data = data_pro.get_predict_data(crossroad_em=(crossroad_id, crossroad_em),
                                             datetime_em=(datetime_stamps,datetime_index_em))

    # 提取数据字典的608个元素中有效的数据
    valid_id = [x for x in data_dict.keys() if len(data_dict[x]) > 0]
    valid_data_dict_em = data_pro.encode_train_test_data(crossroad_em=(crossroad_id, crossroad_em),
                                                        datetime_em=(datetime_stamps, datetime_index_em),
                                                        train_test_data=data_dict,
                                                        )

    model = LSTM_Model(config)
    model.build_model()

    # 在mask状态下启用训练，每隔epoch_per_mask个epoch就随机生成mask，
    # mask作为验证集，保证模型可以对整个数据集充分学习
    for mask_round in range(int(config['training']['epochs'] / config['training']['epoch_per_mask'])):
        print("[Training] Start No. %d mask round." %(mask_round+1))
        # 节点测试集掩码
        masked_id = data_pro.get_dataset_mask(mask_num=35, crossroad_id=valid_id)
        train_id = [x for x in valid_id if x not in masked_id]
        # 在训练集中划分训练和验证集
        training_data_em = {k: v for k, v in valid_data_dict_em.items() if k in train_id}
        val_data_em = {k: v for k, v in valid_data_dict_em.items() if k in masked_id}

        training_data_gen = data_pro.data_generator(training_data_em)
        val_data_gen = data_pro.data_generator(val_data_em)

        model.train_model_generator(training_data_gen, val_data_gen)

        print("[Training] Complete No. %d mask round." %(mask_round+1))
    
    # result = model.predict_submit(predict_data)
    # print(result)



if __name__ == "__main__":
    main()

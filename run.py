from model.baseline import *
from util.data_process import *
import json
import sys
sys.path.append(
    json.load(open('config.json', 'r', encoding='utf-8'))['f_path'])


def main():
    config = json.load(open('config.json', 'r', encoding='utf-8'))
    data_pro = DataProcessor(config=config)
    # 时间编码
    datetime_stamps, datetime_index_em = data_pro.get_datetime_embedding()
    # 从交叉口文件中获取交叉口id和交叉口编码
    crossroad_id, crossroad_em = data_pro.get_crossroads_embedding()
    # 载入已经计算好的车流量数据文件
    predict_data = data_pro.get_predict_data(crossroad_em=(crossroad_id, crossroad_em),
                                             datetime_em=(datetime_stamps, datetime_index_em))
    model = LSTM_Model(config)
    model_file = os.path.join(config['after_training']['well_trained_dir'],
                              config['after_training']['well_trained_model'])
    model.build_model()
    model.load_model(model_file=model_file)

    result = model.predict_submit(predict_data)

    result_loaded = data_pro.load_predict_data()

    data_pro.output_predict_csv(predict_data, result_loaded, datetime_em=(
        datetime_stamps, datetime_index_em))


if __name__ == "__main__":
    main()

from multiprocessing import freeze_support

from get_sen_set import user_input_parse
from data_processing import prepare_company_relation, prepare_dataset, obtain_final_predictions
from company_relation_extraction import prediction_mode
import os

dirname, filename = os.path.split(os.path.abspath(__file__))


def run_extraction(original_news_path, extracted_data_path, input_mode):
    freeze_support()

    file_mode = "default"
    if input_mode != "json":
        file_mode = input_mode
    flag = user_input_parse(input_mode, original_news_path, file_mode)

    if flag:
        prepare_company_relation(dirname + "/data/news_sententces_set.json", input_mode)

        '''
        # If the user choose the training and testing accuracy mode:
        prepare_dataset(divide_label=True)
        train_test_mode()
        '''

        # If the user choose the prediction mode:
        prepare_dataset()
        print(1)
        prediction_mode()
        print(2)
        obtain_final_predictions(dirname + "/data/pre_result.json", extracted_data_path)
        print(3)
        # You can just show the file "final_results.csv" to the front-end.
    else:
        print("None triples extracted!")
    return

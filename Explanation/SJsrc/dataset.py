from Explanation.SJsrc.utils import *

DK_I = "infer"
DK_L = "learn"


class DatasetH:
    def __init__(self, data, train_start_date, train_end_date,
                 valid_start_date, valid_end_date,
                 test_start_date, test_end_date):
        self.data = data
        self.size = len(data)
        self.segments = {
            "train": (train_start_date, train_end_date),
            "valid": (valid_start_date, valid_end_date),
            "test": (test_start_date, test_end_date),
            "explain": (test_start_date, test_end_date)
        }
        self.fetch_orig = True

    def prepare_seg(self, selector, col_set, level, data_key):
        selector = slice(*selector)
        data_df = self.data[data_key]
        data_df = fetch_df_by_col(data_df, col_set)
        data_df = fetch_df_by_index(data_df, selector, level, fetch_orig=self.fetch_orig)
        return data_df

    def prepare(self, segments, col_set, data_key=DK_I):
        level = 'datetime'
        if isinstance(segments, str) and segments in self.segments:
            return self.prepare_seg(self.segments[segments], col_set, level, data_key)

        if isinstance(segments, (list, tuple)) and all(seg in self.segments for seg in segments):
            return [self.prepare_seg(self.segments[seg], col_set, level, data_key) for seg in segments]

    def __len__(self):
        return self.size

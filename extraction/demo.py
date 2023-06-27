import argparse

from com_rel_extract import run_extraction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_news_path", type=str, default=None,
                        help="The path of input text data that you want to load.")
    parser.add_argument("--extracted_data_path", type=str, default=None,
                        help="The path of extracted knowledge that you want to save in.")
    parser.add_argument("--input_format", type=str, default=None,
                        help="The input file format, such as json.")
    args = parser.parse_args()

    run_extraction(args.original_news_path, args.extracted_data_path, args.input_format)
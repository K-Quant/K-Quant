"""
this script is used to update files that are needed in knowledge-empowered models
first step: update the csi300_stock_index.npy
second step[for HIST]: update the csi300_market_value_[start_year]to[end_year].npy
third step[for HIST]: update the csi300_stock2concept.npy
fourth step[for RSR, etc.]: update the csi300_multi_stock2stock_[type].npy
"""

provider_url = "../qlib_data/cn_data/instruments/csi300.txt"
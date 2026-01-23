from analysis.runner import run, run_folders

pulse_path ="/home/augusto/Thesis/master_thesis/tests/data/folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
source_path = "/home/augusto/Thesis/master_thesis/tests/data/folder0/live_data_chip18112025_D1000_B370.mca"
config_path = "/home/augusto/Thesis/master_thesis/resources/config.yaml"
folder_config_path = "/home/augusto/Thesis/master_thesis/resources/folder_config.yaml"

folder = "/home/augusto/Thesis/master_thesis/data/260122/Dscan"
folder0 = "/home/augusto/Thesis/master_thesis/data/260119_trend"
folder1 = "/home/augusto/Thesis/master_thesis/data/260122/Bscan250"


# run(config_path, source_path, pulse_path)
run(config_path, folder0)
# run_folders(config_path, folder0)
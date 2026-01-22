from analysis.runner import run

pulse_path ="/home/augusto/Thesis/master_thesis/tests/data/folder0/live_data_chip18112025_ci5-10-15_hvon.mca"
source_path = "/home/augusto/Thesis/master_thesis/tests/data/folder0/live_data_chip18112025_D1000_B370.mca"
config_path = "/home/augusto/Thesis/master_thesis/resources/config.yaml"
folder_config_path = "/home/augusto/Thesis/master_thesis/resources/folder_config.yaml"

folder = "/home/augusto/Thesis/master_thesis/data/260122/Bscan250"


# run_single(source_path, pulse_path, config_path)
# run_folder(folder_config_path, folder)
run(config_path, folder)
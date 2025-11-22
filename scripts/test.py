from aptapy.plotting import plt

from analysis.fileio import PulsatorFile,SourceFile
from analysis import ANALYSIS_DATA


file = ANALYSIS_DATA / "251118/live_data_chip18112025_D1000_B360.mca"

# pulse_file = PulsatorFile(file)
# model = pulse_file.analyze_pulse(num_sigma=2.5)
source_file = SourceFile(file)
model = source_file.fit_line()
model_forest = source_file.fit_line_forest(num_sigma_left=1.5, num_sigma_right=3.)
# print(model)
print(model_forest)
plt.show()
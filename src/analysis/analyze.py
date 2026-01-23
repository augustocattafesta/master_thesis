# if folder_name == "251118":
#     voltage[i] = np.append(voltage[i], [300, 310, 320])
#     g[i][0] = np.append(g[i][0], unumpy.uarray([33.821245688904206, 40.060014190288435, \
#                                                 47.85847481701872],
#                   [0.004015431655164531, 0.003950437657169777, 0.0038697164947474323]))
# if folder_name == "251127":
#     g_350 = g[i][0][voltage[i] == 350.]
#     g[i][0] = g[i][0][voltage[i] != 350.]
#     voltage[i] = voltage[i][voltage[i] != 350.]
#     voltage[i] = np.append(voltage[i], 350.)
#     g[i][0] = np.append(g[i][0], np.min(g_350))



# derivative_fig = plt.figure("1/G dG/dt")
# tt = np.linspace(*stretched_exp.plotting_range(), 1000)
# yy = stretched_exp_derivative(tt, *stretched_exp.parameter_values()[:3]) / stretched_exp(tt)
# gg = stretched_exp(tt)
# gg /= max(gg)

# plt.plot(gg, yy, label=load_label(folder_name))
# plt.xlabel("Normalized gain")
# plt.ylabel("1/G dG/dt [1/min]")
# plt.tight_layout()
# plt.legend()


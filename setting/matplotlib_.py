import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import LinearLocator, AutoLocator
from matplotlib.dates import DateFormatter
from matplotlib import gridspec, colors


plt.rcParams['figure.figsize'] = ((8/2.54), (6/2.54))
plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.default"] = "rm"
plt.rcParams.update({'font.size': 11})
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1


cmap = ["#e94d5b", "#ef4d28", "#f9a54f", "#25b575",
        "#1bb1e7", "#1477a2", "#a662e5", "#c2f442"]
cmap_m = ["#f4a6ad", "#f6957e", "#fccfa2", "#8de7be",
          "#86d6f2", "#24a9e4", "#b586e0", "#d7f293"]
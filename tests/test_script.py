from bfo import *
import matplotlib.pyplot as plt
from matplotlib import style
style.use('physics')

exp = Experiment('mt12837-1', '/Users/noahwaterfieldprice/Dropbox/Physics/DPhil/projects/bifeo3/data/i16/mt12837-1', 0)
a1 = AverageScan(range(575701, 575705), exp)
a2 = AverageScan(range(575740, 575744), exp)
eta = a1.scans[0].data['eta']
int1 = a1.average('APD')
int2 = a2.average('APD')


fig, ax = plt.subplots()
ax.plot(eta, int1)
ax.plot(eta, int2)
plt.show()


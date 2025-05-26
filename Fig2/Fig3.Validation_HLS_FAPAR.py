import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import math
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.metrics import mean_absolute_error
from scipy import stats
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

fontsize = 18
font = {'family': 'Times New Roman',
            'color': 'k',
            'weight': 'normal',
            'size': fontsize + 2,
            }
cdict=[]

for x in range(0,250):
    hsv=((250-x)/360.0,1,1)
    rgb=colors.hsv_to_rgb(hsv)
    cdict.append(rgb)
    # ax.scatter(100,x,color=(r,0.35,b),marker="s")
cm=colors.ListedColormap(cdict,'zyh')
legendfont = {'family': 'Times New Roman',
                  'size': fontsize}


s = os.sep
# import matplotlib.pyplot as plt
# import numpy as np
i = 1
# filepath="D:\BaiduSyncdisk\Code\GPP\RemoteData\MERRA2"
excelfile = ".\HLS_Field_FAPAR.csv"
excelfile=r"D:\FAPAR_Validation_NA\2024\Fig\FigV5\Fig3\HLS_Field_L30_S30_2.csv"
df = pd.read_csv(excelfile)

fig = plt.figure(figsize=(12, 4), dpi=300)
outdf=pd.DataFrame()
idlist,Nlist,Rlist,SDBlist,RSDBlist,Biaslist,RBiaslist,RMSElist,RRMSElist,URAoptlist,URAtarlist,URAthrlist=[],[],[],[],[],[],[],[],[],[],[],[]
ax = fig.add_subplot(1, 3, 1)
Targetlowxi,Targetlowyi,Targetupxi,Targetupyi,Optimallowxi,Optimallowyi,Optimalupxi,Optimalupyi,Thresholdlowxi,Thresholdlowyi,Thresholdupxi,Thresholdupyi=[],[],[],[],[],[],[],[],[],[],[],[]
for ii in range(120):
    Optimallowxi.append(ii*0.01)
    Optimallowyi.append(ii*0.01*0.95)
    Optimalupxi.append(ii*0.01)
    Optimalupyi.append(ii*0.01*1.05)
    Thresholdlowxi.append(ii*0.01)
    Thresholdlowyi.append(ii*0.01*0.8)
    Thresholdupxi.append(ii*0.01)
    Thresholdupyi.append(ii*0.01*1.2)
    Targetupxi.append(ii * 0.01)
    Targetupyi.append(ii * 0.01 * 1.1)
    Targetlowxi.append(ii * 0.01)
    Targetlowyi.append(ii * 0.01 * 0.9)
# ax.fill_between(Thresholdlowxi, np.array(Thresholdlowyi), np.array(Thresholdupyi), facecolor="dimgray",alpha=1)
# ax.fill_between(Targetlowxi, np.array(Targetlowyi), np.array(Targetupyi), facecolor="darkgray",alpha=1)
# ax.fill_between(Optimallowxi, np.array(Optimallowyi), np.array(Optimalupyi), facecolor="lightgray",alpha=1)
tempdf = df[(df["site"] == "US-Uaf") | (df["site"] == "CA-TP4")]
X=tempdf["FieldFAPAR"].values.tolist()
Y=tempdf["HLSFAPAR"].values.tolist()
Z=tempdf["doy"].values.tolist()

L30tempdf = tempdf[tempdf["sensor"] == "L30"]
L30X=L30tempdf["FieldFAPAR"].values.tolist()
L30Y=L30tempdf["HLSFAPAR"].values.tolist()
L30Z=L30tempdf["doy"].values.tolist()
sc = ax.scatter(L30X, L30Y, s=50, marker="o", c=L30Z, linewidths=0.2,  edgecolors="k",vmin=50,vmax=365, cmap=cm,label="Landsat 8")

S30tempdf = tempdf[tempdf["sensor"] == "S30"]
S30X=S30tempdf["FieldFAPAR"].values.tolist()
S30Y=S30tempdf["HLSFAPAR"].values.tolist()
S30Z=S30tempdf["doy"].values.tolist()
sc = ax.scatter(S30X, S30Y, s=50, marker="d", c=S30Z, linewidths=0.2,  edgecolors="k",vmin=50,vmax=365, cmap=cm)
sc = ax.scatter(1, 1, s=50, marker="d", c=170, linewidths=0.2,  edgecolors="k",vmin=50,vmax=365, cmap=cm,label="Sentinel 2")

ax.legend( loc='lower left', ncol=1, frameon=False, bbox_to_anchor=(-0.05, 0.6),
          handletextpad=0.2,
          labelspacing=0.1, handlelength=1.5, columnspacing=0.5,
          prop=legendfont)

slope, intercept, r_value, p_value, std_err = stats.linregress(Y, X)
r_value = round(r_value, 3)
p_value = round(p_value, 6)
y1 = np.array(Y).reshape(-1, 1)
x1 = np.array(X).reshape(-1, 1)
regr1 = linear_model.LinearRegression()
regr1.fit(x1, y1)
yy1 = regr1.predict(x1)
r21 = round(np.sqrt(regr1.score(x1, y1)), 3)
a1 = round(regr1.coef_[0][0], 3)
b1 = round(regr1.intercept_[0], 3)
rmse1 = round(math.sqrt(mean_squared_error(x1, y1)), 4)
bias1 = round(np.mean(y1 - x1), 4)
SDbias1 = round(np.std(y1 - x1), 4)
meanX=np.mean(X)
RBias=bias1/meanX
RRMSE=rmse1/meanX
RSDB=SDbias1/meanX
N=len(X)
Nlist.append(N)
Rlist.append(r21)
SDBlist.append(SDbias1)
RSDBlist.append(RSDB)
Biaslist.append(bias1)
RBiaslist.append(RBias)
RMSElist.append(rmse1)
RRMSElist.append(RRMSE)
Nopt=0
Ntar=0
Nthr=0
for Ni in range(N):
    # print(Y[Ni], X[Ni])
    if Y[Ni]<=X[Ni]*1.05 and Y[Ni]>=X[Ni]*0.95:
        Nopt=Nopt+1
        Ntar=Ntar+1
        Nthr=Nthr+1
    elif Y[Ni]<=X[Ni]*1.1 and Y[Ni]>=X[Ni]*0.9:
        Ntar = Ntar + 1
        Nthr = Nthr + 1
    elif Y[Ni] <= X[Ni] * 1.2 and Y[Ni] >= X[Ni] * 0.8:
        Nthr = Nthr + 1
URAopt=Nopt/N
URAtar=Ntar/N
URAthr=Nthr/N
URAoptlist.append(URAopt)
URAtarlist.append(URAtar)
URAthrlist.append(URAthr)
idlist.append("EF")

ax.text(0.03, 0.90, "(" + chr(96 + 1) + ") Evergreen Forests", verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize)
ax.text(0.98, 0.33, r'N = ' + str(N), verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
ax.text(0.98, 0.25, r'R = ' + format(abs(r21), '.2f'), verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
if bias1 > 0:
    ax.text(0.98, 0.17, r'Bias = ' + format(abs(bias1), '.2f'), verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
else:
    ax.text(0.98, 0.17, r'Bias = $\mathrm{-}$ ' + format(abs(bias1), '.2f'), verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)

ax.text(0.98, 0.09, r'RMSE = ' + format(abs(rmse1), '.2f'), verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
# ax.text(0.98, 0.15, r'bias = ' + str(bias).ljust(3, '0'), verticalalignment='bottom', horizontalalignment='right',
#         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
if b1 > 0:
    ax.text(0.98, 0.01, r'y = ' + format(abs(a1), '.2f') + "x + " + format(abs(b1), '.2f'), verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
else:
    ax.text(0.98, 0.01, r'y = ' + format(abs(a1), '.2f') + "x $\mathrm{-}$ " + format(abs(b1), '.2f'),
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)

ax.plot(x1, yy1, color='k', linewidth=1.2)
# # y=x线
xxx = [0, 1]
ax.plot(xxx, xxx, color='black', linewidth=0.8, linestyle='--')
# lowxi=[]
# lowyi=[]
# for ii in range(120):
#     if ii>50:
#         lowxi.append(ii*0.01)
#         lowyi.append(ii*0.01*0.9)
#     else:
#         lowxi.append(ii*0.01)
#         lowyi.append(ii*0.01-0.05)
#
# upxi=[]
# upyi=[]
# for ii in range(120):
#     if ii>50:
#         upxi.append(ii*0.01)
#         upyi.append(ii*0.01*1.1)
#     else:
#         upxi.append(ii*0.01)
#         upyi.append(ii*0.01+0.05)
#
# ax.fill_between(lowxi, np.array(lowyi), np.array(upyi), alpha=0.2)
ax.axis([0.1, 1, 0.1, 1])
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
# ax.set(xlim=(0, 1), xticks=np.arange(1, 8),
#        ylim=(0, 56), yticks=np.linspace(0, 56, 9))
ylabel = r"HLS FAPAR"
# xlabel = r"in situ FAPAR"
xlabel = r"${in \ situ}$ FAPAR"

plt.tick_params(labelsize=fontsize)
ax.set_ylabel(ylabel, fontdict=font)
ax.set_xlabel(xlabel, fontdict=font)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(fontsize) for label in labels]
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)


ax = fig.add_subplot(1, 3, 2)
Targetlowxi,Targetlowyi,Targetupxi,Targetupyi,Optimallowxi,Optimallowyi,Optimalupxi,Optimalupyi,Thresholdlowxi,Thresholdlowyi,Thresholdupxi,Thresholdupyi=[],[],[],[],[],[],[],[],[],[],[],[]
for ii in range(120):
    Optimallowxi.append(ii*0.01)
    Optimallowyi.append(ii*0.01*0.95)
    Optimalupxi.append(ii*0.01)
    Optimalupyi.append(ii*0.01*1.05)
    Thresholdlowxi.append(ii*0.01)
    Thresholdlowyi.append(ii*0.01*0.8)
    Thresholdupxi.append(ii*0.01)
    Thresholdupyi.append(ii*0.01*1.2)
    Targetupxi.append(ii * 0.01)
    Targetupyi.append(ii * 0.01 * 1.1)
    Targetlowxi.append(ii * 0.01)
    Targetlowyi.append(ii * 0.01 * 0.9)
    # if ii>50:
    #     Targetupxi.append(ii*0.01)
    #     Targetupyi.append(ii*0.01*1.1)
    #     Targetlowxi.append(ii * 0.01)
    #     Targetlowyi.append(ii * 0.01 * 0.9)
    # else:
    #     Targetupxi.append(ii*0.01)
    #     Targetupyi.append(ii*0.01+0.05)
    #     Targetlowxi.append(ii*0.01)
    #     Targetlowyi.append(ii*0.01-0.05)
# ax.fill_between(Thresholdlowxi, np.array(Thresholdlowyi), np.array(Thresholdupyi), facecolor="dimgray",alpha=1)
# ax.fill_between(Targetlowxi, np.array(Targetlowyi), np.array(Targetupyi), facecolor="darkgray",alpha=1)
# ax.fill_between(Optimallowxi, np.array(Optimallowyi), np.array(Optimalupyi), facecolor="lightgray",alpha=1)

tempdf = df[(df["site"] == "US-HF") | (df["site"] == "CA-TPD") | (df["site"] == "US-Bar")]
L30tempdf = tempdf[tempdf["sensor"] == "L30"]
L30X=L30tempdf["FieldFAPAR"].values.tolist()
L30Y=L30tempdf["HLSFAPAR"].values.tolist()
L30Z=L30tempdf["doy"].values.tolist()
sc = ax.scatter(L30X, L30Y, s=50, marker="o", c=L30Z, linewidths=0.2,  edgecolors="k",vmin=50,vmax=365, cmap=cm)

S30tempdf = tempdf[tempdf["sensor"] == "S30"]
S30X=S30tempdf["FieldFAPAR"].values.tolist()
S30Y=S30tempdf["HLSFAPAR"].values.tolist()
S30Z=S30tempdf["doy"].values.tolist()
sc = ax.scatter(S30X, S30Y, s=50, marker="d", c=S30Z, linewidths=0.2,  edgecolors="k",vmin=50,vmax=365, cmap=cm)



X=tempdf["FieldFAPAR"].values.tolist()
Y=tempdf["HLSFAPAR"].values.tolist()
Z=tempdf["doy"].values.tolist()

ax.text(0.03, 0.9, "(" + chr(96 + 2) + ") Deciduous Forests", verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize)

slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
print(slope, intercept, r_value, p_value, std_err)
r_value = round(r_value, 3)
p_value = round(p_value, 6)
y1 = np.array(Y).reshape(-1, 1)
x1 = np.array(X).reshape(-1, 1)
regr1 = linear_model.LinearRegression()
regr1.fit(x1, y1)
yy1 = regr1.predict(x1)
r21 = round(np.sqrt(regr1.score(x1, y1)), 3)
a1 = round(regr1.coef_[0][0], 3)
b1 = round(regr1.intercept_[0], 3)
rmse1 = round(math.sqrt(mean_squared_error(x1, y1)), 4)
bias1 = round(np.mean(y1 - x1), 4)
SDbias1 = round(np.std(y1 - x1), 4)
meanX=np.mean(X)
RBias=bias1/meanX
RRMSE=rmse1/meanX
RSDB=SDbias1/meanX
N=len(X)
Nlist.append(N)
Rlist.append(r21)
SDBlist.append(SDbias1)
RSDBlist.append(RSDB)
Biaslist.append(bias1)
RBiaslist.append(RBias)
RMSElist.append(rmse1)
RRMSElist.append(RRMSE)
Nopt=0
Ntar=0
Nthr=0
for Ni in range(N):
    # print(Y[Ni], X[Ni])
    if Y[Ni]<=X[Ni]*1.05 and Y[Ni]>=X[Ni]*0.95:
        Nopt=Nopt+1
        Ntar=Ntar+1
        Nthr=Nthr+1
    elif Y[Ni]<=X[Ni]*1.1 and Y[Ni]>=X[Ni]*0.9:
        Ntar = Ntar + 1
        Nthr = Nthr + 1
    elif Y[Ni] <= X[Ni] * 1.2 and Y[Ni] >= X[Ni] * 0.8:
        Nthr = Nthr + 1
URAopt=Nopt/N
URAtar=Ntar/N
URAthr=Nthr/N
URAoptlist.append(URAopt)
URAtarlist.append(URAtar)
URAthrlist.append(URAthr)
idlist.append("DF")
ax.text(0.98, 0.33, r'N = ' + str(N), verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
ax.text(0.98, 0.25, r'R = ' + format(abs(r21), '.2f'), verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
if bias1 > 0:
    ax.text(0.98, 0.17, r'Bias = ' + format(abs(bias1), '.2f'), verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
else:
    ax.text(0.98, 0.17, r'Bias = $\mathrm{-}$ ' + format(abs(bias1), '.2f'), verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)

ax.text(0.98, 0.09, r'RMSE = ' + format(abs(rmse1), '.2f'), verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
# ax.text(0.98, 0.15, r'bias = ' + str(bias).ljust(3, '0'), verticalalignment='bottom', horizontalalignment='right',
#         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
if b1 > 0:
    ax.text(0.98, 0.01, r'y = ' + format(abs(a1), '.2f') + "x + " + format(abs(b1), '.2f'), verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
else:
    ax.text(0.98, 0.01, r'y = ' + format(abs(a1), '.2f') + "x $\mathrm{-}$ " + format(abs(b1), '.2f'),
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)

ax.plot(x1, yy1, color='k', linewidth=1.2)
# # y=x线
xxx = [0, 1]
ax.plot(xxx, xxx, color='black', linewidth=0.8, linestyle='--')


ax.axis([0.1, 1, 0.1, 1])
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
# ax.set(xlim=(0, 1), xticks=np.arange(1, 8),
#        ylim=(0, 56), yticks=np.linspace(0, 56, 9))
# ylabel = r"$\mathrm{APAR_\mathrm{MYD} \ (MJ/m^\mathrm{2}/8d)}$"
# xlabel = r"$\mathrm{EC \ GPP \ (g \ C/m^\mathrm{2}/8d)}$"
ylabel = r"HLS FAPAR"
# xlabel = r"in situ FAPAR"
xlabel = r"${in \ situ}$ FAPAR"

plt.tick_params(labelsize=fontsize)
ax.set_ylabel(ylabel, fontdict=font)
ax.set_xlabel(xlabel, fontdict=font)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(fontsize) for label in labels]
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

ax = fig.add_subplot(1, 3, 3)
Targetlowxi,Targetlowyi,Targetupxi,Targetupyi,Optimallowxi,Optimallowyi,Optimalupxi,Optimalupyi,Thresholdlowxi,Thresholdlowyi,Thresholdupxi,Thresholdupyi=[],[],[],[],[],[],[],[],[],[],[],[]
for ii in range(120):
    Optimallowxi.append(ii*0.01)
    Optimallowyi.append(ii*0.01*0.95)
    Optimalupxi.append(ii*0.01)
    Optimalupyi.append(ii*0.01*1.05)
    Thresholdlowxi.append(ii*0.01)
    Thresholdlowyi.append(ii*0.01*0.8)
    Thresholdupxi.append(ii*0.01)
    Thresholdupyi.append(ii*0.01*1.2)
    Targetupxi.append(ii * 0.01)
    Targetupyi.append(ii * 0.01 * 1.1)
    Targetlowxi.append(ii * 0.01)
    Targetlowyi.append(ii * 0.01 * 0.9)
# ax.fill_between(Thresholdlowxi, np.array(Thresholdlowyi), np.array(Thresholdupyi), facecolor="dimgray",alpha=1)
# ax.fill_between(Targetlowxi, np.array(Targetlowyi), np.array(Targetupyi), facecolor="darkgray",alpha=1)
# ax.fill_between(Optimallowxi, np.array(Optimallowyi), np.array(Optimalupyi), facecolor="lightgray",alpha=1)
# tempdf = df[(df["site"] == "US-HF") | (df["site"] == "CA-TPD") | (df["site"] == "US-HF")]
X=df["FieldFAPAR"].values.tolist()
Y=df["HLSFAPAR"].values.tolist()
Z=df["doy"].values.tolist()

ax.text(0.03, 0.9, "(" + chr(96 + 3) + ") All forest", verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize)

L30tempdf = df[df["sensor"] == "L30"]
L30X=L30tempdf["FieldFAPAR"].values.tolist()
L30Y=L30tempdf["HLSFAPAR"].values.tolist()
L30Z=L30tempdf["doy"].values.tolist()
sc = ax.scatter(L30X, L30Y, s=50, marker="o", c=L30Z, linewidths=0.2,  edgecolors="k",vmin=50,vmax=365, cmap=cm)

S30tempdf = df[df["sensor"] == "S30"]
S30X=S30tempdf["FieldFAPAR"].values.tolist()
S30Y=S30tempdf["HLSFAPAR"].values.tolist()
S30Z=S30tempdf["doy"].values.tolist()
sc = ax.scatter(S30X, S30Y, s=50, marker="d", c=S30Z, linewidths=0.2,  edgecolors="k",vmin=50,vmax=365, cmap=cm)

slope, intercept, r_value, p_value, std_err = stats.linregress(Y, X)
r_value = round(r_value, 3)
p_value = round(p_value, 6)

y1 = np.array(Y).reshape(-1, 1)
x1 = np.array(X).reshape(-1, 1)
regr1 = linear_model.LinearRegression()
regr1.fit(x1, y1)
yy1 = regr1.predict(x1)
r21 = round(np.sqrt(regr1.score(x1, y1)), 3)
a1 = round(regr1.coef_[0][0], 3)
b1 = round(regr1.intercept_[0], 3)
rmse1 = round(math.sqrt(mean_squared_error(x1, y1)), 4)
bias1 = (round(np.mean(y1 - x1), 4))
SDbias1 = round(np.std(y1 - x1), 4)
meanX=np.mean(X)
RBias=bias1/meanX
RRMSE=rmse1/meanX
RSDB=SDbias1/meanX
N=len(X)
Nlist.append(N)
Rlist.append(r21)
SDBlist.append(SDbias1)
RSDBlist.append(RSDB)
Biaslist.append(bias1)
RBiaslist.append(RBias)
RMSElist.append(rmse1)
RRMSElist.append(RRMSE)
Nopt=0
Ntar=0
Nthr=0
for Ni in range(N):
    # print(Y[Ni], X[Ni])
    if Y[Ni]<=X[Ni]*1.05 and Y[Ni]>=X[Ni]*0.95:
        Nopt=Nopt+1
        Ntar=Ntar+1
        Nthr=Nthr+1
    elif Y[Ni]<=X[Ni]*1.1 and Y[Ni]>=X[Ni]*0.9:
        Ntar = Ntar + 1
        Nthr = Nthr + 1
    elif Y[Ni] <= X[Ni] * 1.2 and Y[Ni] >= X[Ni] * 0.8:
        Nthr = Nthr + 1
URAopt=Nopt/N
URAtar=Ntar/N
URAthr=Nthr/N
URAoptlist.append(URAopt)
URAtarlist.append(URAtar)
URAthrlist.append(URAthr)
idlist.append("ALL")
ax.text(0.98, 0.33, r'N = ' + str(N), verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
ax.text(0.98, 0.25, r'R = ' + format(abs(r21), '.2f'), verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
if bias1 > 0:
    ax.text(0.98, 0.17, r'Bias = ' + format(abs(bias1), '.2f'), verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
else:
    ax.text(0.98, 0.17, r'Bias = $\mathrm{-}$ ' + format(abs(bias1), '.2f'), verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)

ax.text(0.98, 0.09, r'RMSE = ' + format(abs(rmse1), '.2f'), verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
# ax.text(0.98, 0.15, r'bias = ' + str(bias).ljust(3, '0'), verticalalignment='bottom', horizontalalignment='right',
#         transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
if b1 > 0:
    ax.text(0.98, 0.01, r'y = ' + format(abs(a1), '.2f') + "x + " + format(abs(b1), '.2f'), verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)
else:
    ax.text(0.98, 0.01, r'y = ' + format(abs(a1), '.2f') + "x $\mathrm{-}$ " + format(abs(b1), '.2f'),
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes, fontname="Times New Roman", color='k', fontsize=fontsize - 2)

ax.plot(x1, yy1, color='k', linewidth=1.2)
# # y=x线
xxx = [0, 1]
ax.plot(xxx, xxx, color='black', linewidth=0.8, linestyle='--')

# lowxi=[]
# lowyi=[]
# for ii in range(120):
#     if ii>50:
#         lowxi.append(ii*0.01)
#         lowyi.append(ii*0.01*0.9)
#     else:
#         lowxi.append(ii*0.01)
#         lowyi.append(ii*0.01-0.05)
#
# upxi=[]
# upyi=[]
# for ii in range(120):
#     if ii>50:
#         upxi.append(ii*0.01)
#         upyi.append(ii*0.01*1.1)
#     else:
#         upxi.append(ii*0.01)
#         upyi.append(ii*0.01+0.05)
#
# ax.fill_between(lowxi, np.array(lowyi), np.array(upyi), alpha=0.2)


#
ax.axis([0.1, 1, 0.1, 1])
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
# ax.set(xlim=(0, 1), xticks=np.arange(1, 8),
#        ylim=(0, 56), yticks=np.linspace(0, 56, 9))
# ylabel = r"$\mathrm{APAR_\mathrm{DAY} \ (MJ/m^\mathrm{2}/8d)}$"
# xlabel = r"$\mathrm{EC \ GPP \ (g \ C/m^\mathrm{2}/8d)}$"
ylabel = r"HLS FAPAR"
xlabel = r"${in \ situ}$ FAPAR"
plt.tick_params(labelsize=fontsize)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
ax.set_ylabel(ylabel, fontdict=font)
ax.set_xlabel(xlabel, fontdict=font)

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(fontsize) for label in labels]
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
#
cbar =plt.colorbar(sc, cax=cax)
labels = cbar.ax.get_xticklabels() + cbar.ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(fontsize) for label in labels]

labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(fontsize) for label in labels]
plt.subplots_adjust(left=0.07, right=0.96, bottom=0.17, top=0.97, wspace=0.3, hspace=0.2)
# plt.savefig("./Fig/"+eachsite+"FAPARins_GPP.png")  # 保存图片
plt.savefig("Fig3.Validation_HLS_FAPAR3.png")  # 保存图片
# idlist,Nlist,Rlist,SDBlist,RSDBlist,Biaslist,RBiaslist,RMSElist,
# RRMSElist,URAoptlist,URAtarlist,URAthrlist=[],[],[],[],[],[],[],[],[],[],[],[]
outdf["ID"]=idlist
outdf["N"]=Nlist
outdf["R"]=Rlist
outdf["SDB"]=SDBlist
outdf["RSDB"]=RSDBlist
outdf["Bias"]=Biaslist
outdf["RBias"]=RBiaslist
outdf["RMSE"]=RMSElist
outdf["RRMSE"]=RRMSElist
outdf["URAopt"]=URAoptlist
outdf["URAtar"]=URAtarlist
outdf["URAthr"]=URAthrlist


outdf.to_csv("Fig3.output2.csv")
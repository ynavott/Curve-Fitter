# region - imports
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy import stats
from scipy.odr import *
# endregion

# matplotlib font
font = {'family': 'Times New Roman',
        'color': 'black',
        'weight': 'normal',
        'size': 16,
        }

# Read .csv file with data and define values
# region
df = pd.read_csv('Data.csv', header=0)
print(df)
x = df.X
dX = df.dX
y = df.Y
dY = df.dY
# endregion

# Initial Guess for parameters
guess = (1, 0)

# Plot Residuals
plot_residuals = False

# R squared
calc_r2_check = True

# Titles
title = ""
x_label = ""
y_label = ""
dY_label = ""

# Plot Style
plot_style = 'seaborn-white'


# region - Curve Fitting Function

def func(p, x):
    a1, a2 = p

    return a1*x + a2
# endregion

# region - ODR

# Degree of freedom


dof = len(x) - len(guess)

model = Model(func)

data = RealData(x, y, sx=dX, sy=dY)

odr = ODR(data, model, beta0=guess)
odr.set_job(fit_type=0)
ODRfit = odr.run()

popt = tuple(ODRfit.beta)
pcov = np.diag(ODRfit.cov_beta)
perr = tuple(math.sqrt(pcov[i]) for i in range(len(pcov)))
redchisq = ODRfit.res_var
chisq = redchisq*dof
pval = stats.chi2.sf(chisq, dof)
# endregion

# region - R squared
if calc_r2_check:
    # Calculate RSS and TSS
    def calc_r2():

        rss = chisq
        tss_i = tuple((y[i]-np.mean(y))**2 for i in range(len(y)))
        tss = np.sum(tss_i)
        R2_value = 1-(rss/tss)
        return R2_value
    R2 = calc_r2()
# endregion

# region - Plotting

# Create subplots
plt.style.use(plot_style)
fig, ax = plt.subplots(figsize=(6.5, 5))

if plot_residuals:
    ax.set_ylabel(dY_label, fontdict=font)
else:
    ax.set_ylabel(y_label, fontdict=font)

xFit = np.linspace(x.iloc[0], x.iloc[-1], num=10000)
y1 = tuple(func(popt, float(xFit[i])) for i in range(len(xFit)))
y2 = tuple(func(popt, float(x[i])) for i in range(len(x)))
if not plot_residuals:
    solution = plt.plot(xFit, y1, 'k', zorder=2)
else:
    solution = plt.plot((x[0], x[len(x)-1]), (0, 0), 'k', zorder=2)
# endregion

# region - Errorbar plots
if plot_residuals:
    dat = plt.errorbar(x, y - y2,
                       xerr=dX, yerr=dY,
                       fmt='k.', ms=4, zorder=0, elinewidth=0.7, capsize=1.5)
else:
    dat = plt.errorbar(x, y,
                       xerr=dX, yerr=dY,
                       fmt='k.', ms=4, zorder=0, elinewidth=0.7, capsize=1.5)
# endregion

# region - Printing
print("Initial Parameter Values: ")
for i in range(len(guess)):
    print(guess[i])

print("Fitted Parameter Values: ")
for i in range(len(popt)):
    print(popt[i], "±", perr[i], "({}%)".format((perr[i]/popt[i])*100))
print("Chi squared (not reduced):", chisq)
print("Degrees of Freedom:", dof)
print("Reduced Chi squared:", redchisq)
print("P-value:", pval)
if calc_r2_check:
    print("R-squared:", R2)
# endregion

# region - Showing

ax.set_xlabel(x_label, fontdict=font)

ax.set_title(title, fontdict=font)

ax.yaxis.label.set_size(12)
ax.xaxis.label.set_size(12)
plt.tight_layout()

plt.show()
# endregion

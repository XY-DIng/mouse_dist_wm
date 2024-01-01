import numpy as np
import scipy as sp
import pandas
import brian2
import copy
import seaborn as sns
from adjustText import adjust_text
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def plotPaperTag(model, paperTag, figureSize, fontSize, saveFig, fileName):
    fig = plt.figure(figsize=figureSize, dpi = 300, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size':fontSize})
    plt.scatter(np.arange(model.n_areas), model.persistentact_df['persistentactBinary'], alpha=0.5, color='r',
                label='Simulation')
    plt.scatter(np.arange(model.n_areas), paperTag, alpha=0.5, label='Summary of experiments')
    plt.legend(bbox_to_anchor=(1.05, 1.15), loc='lower right')
    plt.xticks(np.arange(model.n_areas), model.area_list, rotation='vertical', fontsize=fontSize*0.6)
    plt.yticks([0, 1], ['No delay activity', 'Delay activity'])

    coff = 1 - np.sum(np.logical_xor(model.persistentact_df['persistentactBinary'], paperTag)) / model.n_areas
    plt.title('Simple matching coefficient ' + str(coff))

    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)

    if saveFig:
        fig.savefig('figure/' + fileName, dpi = 300, bbox_inches = 'tight', transparent=True)
    return

def getConnHist(connectivity, xscale, threshold, figureSize, fontSize, saveFig, fileName):
    conn = connectivity
    a = conn[conn != 0]
    b = a.reshape((-1, 1))
    fig = plt.figure(figsize = figureSize, dpi = 300, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': fontSize})

    if xscale == 'log':
        logbins = np.logspace(np.log10(np.min(b)), np.log10(np.max(b)), 20)
        plt.hist(b, bins = logbins)
        plt.xscale('log')
    elif xscale == 'linear':
        plt.hist(b)
    if threshold != 0:
        plt.axvline(threshold, color = 'black')
    plt.xlabel('Connetivity strength(Normalized)')
    plt.ylabel('Frequency')
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    if saveFig:
        fig.savefig('figure/' + fileName, dpi = 300, bbox_inches = 'tight', transparent=True)
    return

def getNnodeLoop(conn, showFig, area_list, figureSize, fontSize, saveFig, fileName, connectivityNorm=False):
    if showFig:
        plt.figure()
        plt.imshow(conn)
        plt.colorbar()
    cycDict = {}
    for i in [2, 3]:
        if i == 2:
            c = np.power(conn, 1/i)
            curC = np.dot(c, c)
        if i == 3:
            c = np.power(conn, 1/i)
            curC = np.dot(np.dot(c, c), c)
        if showFig:
            plt.figure()
            plt.imshow(curC)
            plt.colorbar()
        b = np.diag(curC)
        if showFig:
            print(b)
        # b = np.power(b, 1/i)
        if showFig:
            fig = plt.figure(figsize=figureSize, dpi=300, facecolor='w', edgecolor='k')
            plt.rcParams.update({'font.size': fontSize})
            plt.plot(b)
            plt.title(str(i) + '-node loop')
            plt.xticks(range(len(area_list)), area_list, rotation=90, fontsize=fontSize*0.6)
            plt.ylabel('sum of cycle strength for each area')
            axes = plt.gca()
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            if saveFig:
                fig.savefig('figure/' + fileName[i] , dpi=300, bbox_inches='tight', transparent=True)
        cycDict[i] = b


    return cycDict



def adjust_legend_position(ax, legend, iterations=10):
    # Get the bounding box of the original plot
    ax_bb = ax.get_position()
    for _ in range(iterations):
        # Get the bounding boxes of the legend and plot
        legend_bb = legend.get_window_extent()
        ax_bb_data_coords = ax.transAxes.inverted().transform(legend_bb)
        
        overlaps = False
        for line in ax.get_lines():
            xdata, ydata = line.get_data()
            for x, y in zip(xdata, ydata):
                if ax_bb_data_coords.contains(x, y):
                    overlaps = True
                    break
            if overlaps:
                break
        
        # If the legend overlaps with the data, adjust its position
        if overlaps:
            anchor = legend.get_bbox_to_anchor().get_points()
            anchor[1][1] += 0.05  # move the legend down
            legend.set_bbox_to_anchor(anchor)
        else:
            break


def plotXYcompr(X, Y, Xlabel, Ylabel, area_list, figureSize, fontSize, saveFig, fileName):
    # key test PV vs. FR pattern

    Xnorm = X / np.abs(X).max()
    Ynorm = Y / np.abs(Y).max()

    fig, ax = plt.subplots(figsize=figureSize, dpi=300, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': fontSize})
    ax.plot(Xnorm, label=Xlabel)
    ax.plot(Ynorm, label=Ylabel)
    ax.set_xlabel('Area')
    ax.set_ylabel('Normalized value')
    ax.set_xticks(range(len(X)))
    ax.set_xticklabels(area_list, rotation=90, fontsize=fontSize*0.6)
    
    # Place the legend at the top left, outside of the main plot
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.12), borderaxespad=0.,  fontsize=fontSize*0.75, frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if saveFig:
        fig.savefig('figure/' + fileName, dpi=300, bbox_inches='tight', transparent=True)

#  def plotXYcompr(X, Y, Xlabel, Ylabel, area_list, figureSize, fontSize, saveFig, fileName):
#     # key test  PV vs. FR pattern

#     Xnorm = X / np.abs(X).max()
#     Ynorm = Y / np.abs(Y).max()

#     fig = plt.figure(figsize=figureSize, dpi = 300, facecolor='w', edgecolor='k')
#     plt.rcParams.update({'font.size': fontSize})
#     plt.plot(Xnorm, label=Xlabel)
#     plt.plot(Ynorm, label=Ylabel)
#     # plt.plot(X / X.max(), label=Xlabel)
#     # plt.plot(Y / Y.max(), label=Ylabel)
#     plt.xlabel('Area')
#     plt.ylabel('Normalized value')
#     plt.xticks(range(len(X)), area_list, rotation=90, fontsize = fontSize*0.6)
#     plt.legend(bbox_to_anchor=(1.05, 1.15), loc='lower right')

#     axes = plt.gca()
#     axes.spines['top'].set_visible(False)
#     axes.spines['right'].set_visible(False)
#     if saveFig:
#         fig.savefig('figure/' + fileName, dpi = 300, bbox_inches = 'tight', transparent=True)

def plotXYcomprCorr(X, Y, Xlabel, Ylabel, parameters, showLabel, area_list, frThreshold, figureSize, fontSize, saveFig, fileName):
    # key test  PV vs. FR pattern

    Ycopy = copy.copy(Y)
    Xnorm = X / np.abs(X).max()
    Ynorm = Y / np.abs(Y).max()
    Xnorm = Xnorm[Ycopy >= frThreshold]
    Ynorm = Ynorm[Ycopy >= frThreshold]

    div_name_list, div_color_list = parameters['div_name_list'], parameters['div_color_list']
    div = parameters['div']

    fig = plt.figure(figsize=figureSize, dpi=300, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': fontSize})
    g1 = sns.regplot(Xnorm, Ynorm)
    plt.scatter(Xnorm, Ynorm)

    ax = plt.gca()
    if showLabel:
        texts = []
        for i in range(len(area_list)):
            acr = area_list[i]
            for div_name, div_color in zip(div_name_list, div_color_list):
                if acr in div[div_name]:
                    texts += [ax.text(Xnorm[i], Ynorm[i], acr,
                                      color=div_color, fontsize=fontSize*0.4)]

        # # use adjust library to adjust the position of annotations.
        if True:
            adjust_text(texts, Xnorm, Ynorm,
                        ax=ax, precision=0.001,
                        arrowprops=dict(arrowstyle='-', color='gray', alpha=.8))

    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.xlim([min(Xnorm) - 0.05, max(Xnorm) + 0.05])
    r, pvalue = sp.stats.pearsonr(Xnorm, Ynorm)
    rS, pvalueS = sp.stats.spearmanr(Xnorm, Ynorm)
    print(['r', r, 'r^2', r**2, 'pvalue', pvalue, 'rSpear', rS, 'rSpear^2', rS ** 2,'pvalueSpear', pvalueS])

    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    if saveFig:
        fig.savefig('figure/' + fileName, dpi=300, bbox_inches='tight', transparent=True)
    return {'r': r, 'r^2': r**2, 'pvalue': pvalue, 'rSpear': rS, 'rSpear^2': rS ** 2,'pvalueSpear': pvalueS}

def plotXYcomprCorrNotNormalize(X, Y, Xlabel, Ylabel, parameters, showLabel, area_list, frThreshold, figureSize, fontSize, saveFig, fileName):
    # key test  PV vs. FR pattern

    Ycopy = copy.copy(Y)
    X = X[Ycopy >= frThreshold]
    Y = Y[Ycopy >= frThreshold]

    div_name_list, div_color_list = parameters['div_name_list'], parameters['div_color_list']
    div = parameters['div']

    fig = plt.figure(figsize=figureSize, dpi=300, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': fontSize})
    g1 = sns.regplot(X, Y)
    plt.scatter(X, Y)

    ax = plt.gca()
    if showLabel:
        texts = []
        for i in range(len(area_list)):
            acr = area_list[i]
            for div_name, div_color in zip(div_name_list, div_color_list):
                if acr in div[div_name]:
                    texts += [ax.text(X[i], Y[i], acr,
                                      color=div_color, fontsize=fontSize*0.4)]

        # # use adjust library to adjust the position of annotations.
        if True:
            adjust_text(texts, X, Y,
                        ax=ax, precision=0.001,
                        arrowprops=dict(arrowstyle='-', color='gray', alpha=.8))

    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.xlim([min(X) - 0.05, max(X) + 0.05])
    r, pvalue = sp.stats.pearsonr(X, Y)
    rS, pvalueS = sp.stats.spearmanr(X, Y)
    print(['r', r, 'r^2', r**2, 'pvalue', pvalue, 'rSpear', rS, 'rSpear^2', rS ** 2,'pvalueSpear', pvalueS])

    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    if saveFig:
        fig.savefig('figure/' + fileName, dpi=300, bbox_inches='tight', transparent=True)
    return {'r': r, 'r^2': r**2, 'pvalue': pvalue, 'rSpear': rS, 'rSpear^2': rS ** 2,'pvalueSpear': pvalueS}


def plotXYcomprResidual(X, Y, Xlabel, frThreshold, figureSize, fontSize, saveFig, fileName):
    # key test  PV vs. FR pattern
    Ycopy = copy.copy(Y)
    Xnorm = X / X.max()
    Ynorm = Y / Y.max()
    Xnorm = Xnorm[Ycopy >= frThreshold]
    Ynorm = Ynorm[Ycopy >= frThreshold]

    fig = plt.figure(figsize=figureSize, dpi=300, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': fontSize})
    g2 = sns.residplot(Xnorm, Ynorm)
    plt.ylim([-0.5, 0.5])
    plt.xlabel(Xlabel)
    plt.ylabel('Residual')

    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    if saveFig:
        fig.savefig('figure/' + fileName, dpi=300, bbox_inches='tight', transparent=True)


def plotXYLogReg(X, Y, Xlabel, Ylabel, showLabel, parameters, area_list, noLabelAreas, frThreshold, figureSize,
                 fontSize, saveFig, fileName):
    div_name_list, div_color_list = parameters['div_name_list'], parameters['div_color_list']
    div = parameters['div']

    Ycopy = copy.copy(Y)
    Xnorm = X / X.max()
    X2d = Xnorm.reshape((-1, 1))
    Ynorm = np.array([int(x) for x in Ycopy >= frThreshold])
    print(Ynorm)

        # Sample data
    # X is your feature matrix, y is your target vector
    # For instance, X might be the ages, and y might be binary labels (1 for High Wealth, 0 for Low Wealth)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scaling the features (especially important if you have multiple features of different scales)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Training a logistic regression model
    clf = LogisticRegression()
    clf.fit(X_train_scaled, y_train)

    # Predicting on the test set
    y_pred = clf.predict(X_test_scaled)

    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")


    #     logR = LogisticRegression(solver='liblinear', random_state=77).fit(X2d, Ynorm)
    #     c1, c0 = logR.coef_[0][0], logR.intercept_[0]
    #     print(c1, c0, logR.score(X2d, Ynorm), logR)

    exog, endog = sm.add_constant(X2d), Ynorm
    logRmodel = sm.GLM(endog, exog, family=sm.families.Binomial(link=sm.families.links.logit))
    # link function is logit, the inverse of sigmoid. Xb = g(u). g is link function. b is independent varible. u is dependent varible.
    logR = logRmodel.fit()
    #     print(logR.summary(), logR.params)
    
    # roc_auc_score(Ynorm, logR.predict(exog))

    fig = plt.figure(figsize=figureSize, dpi=300, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': fontSize})
    g1 = sns.regplot(Xnorm, Ynorm, logistic=True, ci=None, scatter=False)
    plt.scatter(Xnorm, Ynorm, alpha=0.3, edgecolor='none')

    ax = plt.gca()
    if showLabel:
        texts = []
        xPos = []
        yPos = []
        for i in range(len(area_list)):
            acr = area_list[i]
            for div_name, div_color in zip(div_name_list, div_color_list):
                if acr in div[div_name] and acr not in noLabelAreas:
                    xPos += Xnorm[i]
                    yPos += Ynorm[i]
                    texts += [ax.text(Xnorm[i], Ynorm[i], acr,
                                      color=div_color, fontsize=fontSize * 0.4)]

        # # use adjust library to adjust the position of annotations.
        if True:
            adjust_text(texts, Xnorm, Ynorm,
                        ax=ax, precision=0.001,
                        arrowprops=dict(arrowstyle='-', color='gray', alpha=.8))

    c1, c0 = logR.params[1], logR.params[0]
    x = c1 * Xnorm + c0
    yPred = 1 / (1 + np.exp(-x)) >= 0.5
    yScore = sum(yPred == Ynorm) / len(Ynorm)


    #     plt.scatter(Xnorm, 1 / (1 + np.exp(-k)))  # TODO chekc more
    #     plt.scatter(Xnorm, yPred)

    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.xlim([min(Xnorm) - 0.05, max(Xnorm) + 0.05])

    print(['prediction score', yScore])

    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    if saveFig:
        fig.savefig('figure/' + fileName, dpi=300, bbox_inches='tight', transparent=True)
    return {'prediction score': yScore}


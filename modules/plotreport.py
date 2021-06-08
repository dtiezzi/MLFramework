import scikitplot as skplt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
import datetime

def plotgraphs(y_test, y_pred, prob_pos, name, pjname, normalizeCM=False, plot_calibration=True, title="", pos_label=1, saveplots=0):
    dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=normalizeCM, title=name)
    plt.savefig('./reports/graphs/' + pjname + "_" + name + "_CM_" + dt + '.png') if saveplots else 0
    skplt.metrics.plot_roc(y_test, prob_pos, plot_micro=False, plot_macro=False, classes_to_plot=[1], title=name,figsize=(10,10))
    plt.savefig('./reports/graphs/' + pjname + "_" + name + "_ROC_" + dt + '.png') if saveplots else 0

    prob_pos = prob_pos[:,1]
    clf_score = brier_score_loss(y_test, prob_pos, pos_label=pos_label)
    with open('./reports/txtfiles/' + pjname + "_" + name + "_" + dt + '.txt', 'a') as f:
        print("%s:" % name, file=f)
        print("\tBrier: %1.3f" % (clf_score), file=f)
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred), file=f)
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred), file=f)
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred), file=f)

    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
    plt.rcParams.update({'font.size': 22})
    plt.rc('legend',**{'fontsize':22})
    fig = plt.figure(3, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated",)
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label="%s (%1.3f)" % (name, clf_score))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                    histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration Plot  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("count")
    ax2.legend(loc="upper center", ncol=2)

    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
            ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(22)
        
    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
            ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(22)

    plt.tight_layout()
    plt.savefig('./reports/graphs/' + pjname + "_" + name + "_CP_" + dt + '.png') if saveplots else 0
    plt.show()

# def plotTree(viz):
#     viz.save('./reports/graphs/' + name + "_TREE_" + dt + '.svg')
from modules.readDb import ReadDb
from modules.resample import Resampling
from modules.runmodel import Run
from modules.model import Mlmodel
from modules.funcs import printMloptions, printdf, printFiles, plotTree, pjname
from modules.gridsc import Grid
from modules.models import Modelist
import os
import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--input', required=True, help='CSV file with columns as attributes and rows as features')
# args = ap.parse_args()

files = os.listdir('./static')
filesList = ["./static/" + f for f in files if f.endswith('.csv')]

def main():
    project_name = pjname()
    fn = printFiles(filesList)
    db = ReadDb(filesList[fn])
    db.preprocess()
    printdf(db.X, "Independent Variables Summary")
    tt = Resampling(db.X, db.y)
    tt.smote()
    op = printMloptions()
    grid = input('Grid Search(y/n)?').lower()
    if grid == 'y':
        gs = Grid()
        ml = gs.runGrid(op, tt.X_train, tt.y_train)
        #print(ml.modelist)
        m = Mlmodel(op)
        m.bildmodels(ml)
        for model, name in zip(m.models, m.names):
            runModel = Run(model, tt.X_train, tt.y_train, tt.X_test, tt.y_test)
            runModel.runModel(name=name, pjname=project_name)
            runModel.reportpred(title=name, pjname=project_name)
            if name == 'Decision Tree':
                viz = plotTree(runModel.fittedModel, tt.X_train, tt.X_test, tt.y_train, target=db.target, class_nm=db.class_name)
                viz.save('./reports/graphs/' + project_name + "_decision_tree.svg")
            #runModel.report()
    else:
        m = Mlmodel(op)
        m.bildmodels(Modelist())
        for model, name in zip(m.models, m.names):
            runModel = Run(model, tt.X_train, tt.y_train, tt.X_test, tt.y_test)
            runModel.runModel(name=name, pjname=project_name)
            runModel.reportpred(title=name, pjname=project_name)
            if name == 'Decision Tree':
                viz = plotTree(runModel.fittedModel, tt.X_train, tt.X_test, tt.y_train, target=db.target, class_nm=db.class_name)
                viz.save('./reports/graphs/' + project_name + "_decision_tree.svg")
            #runModel.report()


if __name__ == '__main__':
    main()
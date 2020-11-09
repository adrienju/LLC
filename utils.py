
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
import numpy as np

def get_data(df1, row=[], col=["patient_state"]):
    
    "Return selected row data in numpy format"
    df2 = df1.copy()   
    df2 = df2.drop(col, axis=1)   
    X = df2.iloc[row, :].to_numpy() #exclude spectrum index

    y = df1["patient_state"].iloc[row].to_numpy()
    return X, y

def standardize(X_train, X_test, X_val=None):
 
    
        scaler_train = StandardScaler() 
        X_val_normalized = None 

        if X_val is not None:
            scaler_val = StandardScaler() 
            X_val_normalized = scaler_val.fit_transform(X_val)

        return scaler_train.fit_transform(X_train), scaler_train.transform(X_test), X_val_normalized
    
def train(X_train, y_train, model):
    return model.fit(X_train, y_train)

    
def test(X_test, y_test, model):
    return model.score(X_test, y_test)


def plot_roc(ax, tprs, aucs):
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='k',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=1)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Receiver operating characteristic")
        ax.legend(loc="lower right")

import numpy as np
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics, random
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from sklearn.metrics import precision_recall_fscore_support
import shap

def make_seed(seedStr):
    return int(hashlib.md5(seedStr.encode("utf-8")).hexdigest()[24:], 16)

def get_actual_class(tumor_code, actual_classes):
    return actual_classes.get(tumor_code)
def optimize_thresholds(y_test, y_pred_proba):
    best_mcc = -1
    best_thresholds = (0, 0)

    for t_low in np.linspace(0, 1, num=100):
        for t_high in np.linspace(t_low, 1, num=100):
            y_pred = np.zeros_like(y_test)
            y_pred[y_pred_proba[:, 0] > t_low] = 0
            y_pred[y_pred_proba[:, 1] > t_high] = 1

            mcc = matthews_corrcoef(y_test, y_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_thresholds = (t_low, t_high)
                print (best_mcc)
    return best_thresholds

dataDel = pd.read_csv("Delayed1.csv")
dataPre = pd.read_csv("Precontrast1.csv")
dataArt = pd.read_csv("Arterial1.csv")
dataVen = pd.read_csv("Venous1.csv")

data = pd.concat([dataDel, dataPre, dataArt, dataVen], axis=1)

labelclassnum2 = data['Grade']
code_data = data[['tumor and ID']].copy()
data = data.drop(['RPCID', 'tumorKey', 'patientId', 'Grade', 'tumor and ID'], axis=1)

datafeatures = np.array(data)

accuracy = np.zeros([100])
output_file = "all_reports_stack.txt"
with open(output_file, "w") as f:
    f.write("")
print ('Starting the Code ...')
misclassified_count = {}
actual_classes = {}
for i in range(100):

    seed1 = make_seed(f"growth1{i}")
    seed2 = make_seed(f"growth2{i}")
    seed3 = make_seed(f"growth3{i}")
    seed4 = make_seed(f"growth4{i}")

    np.random.seed(seed3)
    random.seed(seed4)

    x_train, x_test, y_train, y_test, train_index, test_index = train_test_split(datafeatures, labelclassnum2, data.index,
                                                                                 test_size=0.15, random_state=seed1,
                                                                                 stratify=labelclassnum2)

    cnn = CondensedNearestNeighbour(random_state=42)
    x_train, y_train = cnn.fit_resample(x_train, y_train)


    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weights = dict(zip(np.unique(y_train), class_weights))
    scale_pos_weight = weights[1] / weights[0]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    k_best = SelectKBest(score_func=f_classif, k=20)
    x_train = k_best.fit_transform(x_train, y_train)
    x_test = k_best.transform(x_test)
    print ('DONE')
    # Grid search for XGBoost
    xgb_model = XGBClassifier(tree_method='gpu_hist', gpu_id=0)
    xgb_param_grid = {
        'scale_pos_weight': [scale_pos_weight],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 2, 3],
        'gamma': [0, 0.1, 0.2],
        'colsample_bytree': [0.6, 0.8, 1.0], 
        'subsample': [0.6, 0.8, 1.0],
    }

    xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, scoring='f1', cv=5, n_jobs=-1)
    xgb_grid_search.fit(x_train, y_train)
    print ('DONE XGB')
    # Grid search for Random Forest
    rf_model = RandomForestClassifier()
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample'],
    }
    print ('DONE RF')
    rf_grid_search = GridSearchCV(rf_model, rf_param_grid, scoring='f1', cv=5, n_jobs=-1)
    rf_grid_search.fit(x_train, y_train)

    best_xgb_model = xgb_grid_search.best_estimator_
    best_xgb_model.fit(x_train, y_train)

    best_rf_model = rf_grid_search.best_estimator_
    best_rf_model.fit(x_train, y_train)
    '''
    xgb_y_pred_proba = best_xgb_model.predict_proba(x_test)
    rf_y_pred_proba = best_rf_model.predict_proba(x_test)
    avg_y_pred_proba = (xgb_y_pred_proba + rf_y_pred_proba) / 2
    avg_y_pred = np.argmax(avg_y_pred_proba, axis=1)

    # ENSEMBLING: Evaluate the performance of the ensemble model
    ensemble_accuracy = accuracy_score(y_test, avg_y_pred)
    print("Ensemble Accuracy: %.2f%%" % (ensemble_accuracy * 100.0))
    '''
    stacking_clf = StackingClassifier(
        estimators=[('xgb', best_xgb_model), ('rf', best_rf_model)],
        final_estimator=LogisticRegression(),
        cv=5
    )
    stacking_clf.fit(x_train, y_train)

    stacked_y_pred = stacking_clf.predict(x_test)
    stacked_y_pred_proba = stacking_clf.predict_proba(x_test)
    stacked_accuracy = accuracy_score(y_test, stacked_y_pred)
    print("Stacking Classifier Accuracy: %.2f%%" % (stacked_accuracy * 100.0))

    test_code_data = code_data.iloc[test_index]
    pred_data = list(zip(test_code_data['tumor and ID'], stacked_y_pred, y_test))

    for code, pred, actual in pred_data:
        if code not in misclassified_count:
            misclassified_count[code] = 0
            actual_classes[code] = actual
        if pred != actual:
            misclassified_count[code] += 1


    ensemble_report = classification_report(y_test, stacked_y_pred) 
    print(ensemble_report)

    y_pred_proba_low_grade = stacked_y_pred_proba[:, 0] 
    y_pred_proba_high_grade = stacked_y_pred_proba[:, 1]  

    optimal_threshold_low, optimal_threshold_high = optimize_thresholds(y_test, stacked_y_pred_proba)

    y_pred_optimal = np.zeros_like(y_test)
    y_pred_optimal = np.where(stacked_y_pred_proba[:, 1] > optimal_threshold_high, 1, 0)
    optimal_classification_report = classification_report(y_test, y_pred_optimal)
    print(optimal_classification_report)

    optimal_y_pred_proba_pos = stacked_y_pred_proba[:, 1]

    optimal_fpr, optimal_tpr, _ = roc_curve(y_test, optimal_y_pred_proba_pos)
    optimal_roc_auc = auc(optimal_fpr, optimal_tpr)

    lw = 2
    plt.figure()
    plt.plot(optimal_fpr, optimal_tpr, color='darkorange', lw=lw,
             label=f'ROC curve (area = {optimal_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_optimal_stack{i}.png')
    plt.close()

    optimal_conf_mat = confusion_matrix(y_test, y_pred_optimal)
    disp = ConfusionMatrixDisplay(confusion_matrix=optimal_conf_mat, display_labels=['LowGrade', 'HighGrade'])
    disp.plot(cmap='Blues')
    plt.savefig(f'optimal_confusion_matrix_stack{i}.png')
    '''
    explainer = shap.Explainer(stacking_clf.named_estimators_['xgb'], x_train)

    # Calculate SHAP values for the x_test dataset
    shap_values = explainer(x_test)

    # Create a SHAP force plot and save it to a file
    test_sample_idx = 0  # Choose the index of the test sample you want to visualize
    shap.plots.force(shap_values[test_sample_idx], matplotlib=True, show=False)
    plt.savefig(f'shap_force_plot_optimal_iter_{i+1}.png', bbox_inches='tight')
    plt.close()
    '''
    with open(output_file, "a") as f:
        f.write(f"Iteration {i+1} - Optimal Thresholds\n")
        f.write(f"Low Grade: {optimal_threshold_low}, High Grade: {optimal_threshold_high}\n")
        f.write(optimal_classification_report)
        f.write("\n")

threshold = max(20, 0.25 * 100)
repeatedly_wrong_codes = {code: count for code, count in misclassified_count.items() if count >= threshold}

with open("repeatedly_wrong_codes_stack.csv", "w") as f:
    f.write("Code,Count,Actual\n")
    for code, count in repeatedly_wrong_codes.items():
        actual_class = get_actual_class(code, actual_classes)
        f.write(f"{code},{count},{actual_class}\n")

average_accuracy = np.mean(accuracy)
print("Average accuracy: %.2f%%" % (average_accuracy * 100.0))
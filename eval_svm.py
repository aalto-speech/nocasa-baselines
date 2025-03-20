import pandas as pd
import opensmile
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os.path
import pickle

clf= None
#load the SVM model
with open('svm_model.pkl','rb') as f:
    clf = pickle.load(f)
    
test = pd.read_csv("test.csv")
feats = []
fnames = []
print("Preparing test data")
if os.path.isfile("test_compare_feats.csv"):
    test_feats = pd.read_csv("test_compare_feats.csv")
    for audio in tqdm(test["File Name"]):
        fnames.append(audio)
else: 
    for audio in tqdm(test["File Name"]):
        data = smile.process_file("test/"+audio)
        fnames.append(audio)
        feats += [data]
    test_feats = pd.concat(feats).reset_index(drop=True)
    test_feats.to_csv("test_compare_feats.csv", index = False)
print("Evaluating SVC model")
y_pred = clf.predict(test_feats.values)
predicted = {"File Name": fnames, "predicted": y_pred}
predicted = pd.DataFrame(data=predicted)
predicted.to_csv("test_predicted.csv", index = False)

import pandas as pd
import opensmile
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os.path
import pickle

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
)

train = pd.read_csv("train.csv")
feats = []
labels = []
print("Preparing training data")
if os.path.isfile("train_compare_feats.csv"):
    train_feats = pd.read_csv("train_compare_feats.csv")
    for audio in tqdm(train["File Name"]):
        labels.append(train[train["File Name"]==audio]["Score"].values[0])
else:
    for audio in tqdm(train["File Name"]):
        data = smile.process_file("train/"+audio)
        feats += [data]
        labels.append(train[train["File Name"]==audio]["Score"])
    train_feats = pd.concat(feats).reset_index(drop=True)
    train_feats.to_csv("train_compare_feats.csv", index = False)

print("Training SVC model (could take a few minutes)")
clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, verbose=True))
clf.fit(train_feats.values, labels)

#save the model
with open('svm_model.pkl','wb') as f:
    pickle.dump(clf,f)
    
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

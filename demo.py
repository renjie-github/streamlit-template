import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

st.title("Streamlit Demo")

st.write("""
# Comparison of different classifiers
""")

# Select which dataset and which kind of classification algorithms to use
dataset_name = st.sidebar.selectbox(
    "Select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))

classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("KNN", "SVM", "Random Forest"))

visualization_method = st.sidebar.selectbox(
    "Select Dimension Reduction Method", ("PCA", "TSNE"))


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target
    return X, y


X, y = get_dataset(dataset_name)
st.write(f"Shape of dataset: {X.shape}")
st.write(f"Number of classes: {len(np.unique(y))}")

# Set paramester for selected model


def add_parameter_ui(clf_name, vis_method):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 50)
        random_state = st.sidebar.slider("random_state", 0, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["random_state"] = random_state

    dim_reducer = PCA(2) if vis_method == "PCA" else TSNE(2)
    return params, dim_reducer


params, dim_reducer = add_parameter_ui(classifier_name, visualization_method)

# Load classifier and set its parameters


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"],
                                     random_state=params["random_state"])
    return clf


clf = get_classifier(classifier_name, params)

# Train this classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1)
clf.fit(X_train, y_train)
y_val_pred = clf.predict(X_val)
y_pred = clf.predict(X_test)

acc_val = accuracy_score(y_val, y_val_pred)
acc_test = accuracy_score(y_test, y_pred)
st.write(f"classifier: {classifier_name}")
st.write(
    f"----accuracy on validation set: {acc_val :.2f}")
st.write(
    f"----accuracy on test set: {acc_test :.2f}")


# Dimension reduction and plot the data
X_projected = dim_reducer.fit_transform(X)
fig, ax = plt.subplots()
plot_fig = ax.scatter(X_projected[:, 0], X_projected[:, 1],
                      c=y, alpha=0.8, cmap="viridis")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_title("Data Visualization")
plt.colorbar(plot_fig)
st.pyplot(fig)

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTEN
import re
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



def my_func(loc):
    result = re.findall(r"\s+([A-Z]{2})$", loc)
    if len(result) == 1:
        return result[0][1:]
    else:
        return loc


data = pd.read_excel("final_project.ods", engine="odf", dtype=str)
data = data.fillna("")
data["location"] = data["location"].apply(my_func)
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

ros = SMOTEN(
    random_state=42,
    k_neighbors=2,
    sampling_strategy={
        "director_business_unit_leader": 200,
        "specialist": 100,
        "managing_director_small_medium_company": 50,
    },
)



x_train, y_train = ros.fit_resample(x_train, y_train)



preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1)), "title"),
        ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
        (
            "description",
            TfidfVectorizer(
                stop_words=["english"], ngram_range=(1, 1), min_df=0.01, max_df=0.95
            ),
            "description",
        ),
        ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
        ("industry", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 2)), "industry"),
    ]
)

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LogisticRegression(
    max_iter=5000,
    class_weight="balanced"
)),
])

# ------------------ GRID SEARCH ------------------

param_grid = {
    "regressor__C": [0.01, 0.1, 1, 10],
    "regressor__penalty": ["l2"],
    "regressor__class_weight": [None, "balanced"]
}
grid = GridSearchCV(
    cls,
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2
)

grid.fit(x_train, y_train)

best_model = grid.best_estimator_

y_predict = best_model.predict(x_test)

print("Best params:", grid.best_params_)
print(classification_report(y_test, y_predict))
# ------------------ VISUALIZATION ------------------

# Confusion Matrix
cm = confusion_matrix(y_test, y_predict)

plt.figure(figsize=(10,8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=best_model.classes_,
    yticklabels=best_model.classes_
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()


# F1-score visualization
report = classification_report(y_test, y_predict, output_dict=True)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(10,6))
sns.barplot(
    x=df_report.index[:-3],
    y=df_report["f1-score"][:-3]
)

plt.xticks(rotation=45)
plt.ylabel("F1 Score")
plt.title("F1-score per Class")
plt.show()


# Class distribution after SMOTE
plt.figure(figsize=(8,5))
sns.countplot(x=y_train)

plt.xticks(rotation=45)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Career Level")
plt.ylabel("Count")
plt.show()

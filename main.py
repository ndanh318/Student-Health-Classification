import dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def replace_data(loc):
    value = "All of these"
    item = loc.split(";")
    if item[-1] == value:
        return value
    else:
        return loc


if __name__ == '__main__':
    URL = "https://s.net.vn/bsQu"
    dataset_path = dataset.download_and_unzip(URL)
    data = pd.read_csv(dataset_path)
    data = data.drop("Names", axis=1)

    # visualize data
    plt.figure(figsize=(8, 6))
    for i, col in enumerate(data.columns):
        sns.countplot(data=data[[col]], x=col)
        plt.title(f'Countplot of {col}')
        plt.xlabel(col)
        plt.ylabel("Count")

        plt.show()

    data = data.drop("Mobile Phone ", axis=1)
    data = data.dropna()
    data["Mobile phone activities"] = data["Mobile phone activities"].apply(replace_data)
    data["Usage symptoms"] = data["Usage symptoms"].apply(replace_data)

    health_rate_dict = {
        "Excellent;Good": "Excellent",
        "Good;Fair": "Fair",
        "Excellent;Good;Fair;Poor": "Excellent"
    }
    data = data.replace({"Health rating": health_rate_dict})

    # split data
    target = "Health rating"
    x = data.drop(target, axis=1)
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # preprocessing
    nominal = ["Mobile phone activities",
               "Educational Apps",
               "Usage distraction",
               "Useful features",
               "Health Risks",
               "Beneficial subject",
               "Usage symptoms",
               "Health precautions"]
    ordinal = x_train.columns.drop(nominal).tolist()

    age = ["16-20", "21-25", "26-30", "31-35"]
    gender = x_train["Gender "].unique()
    mobile_system = x_train["Mobile Operating System "].unique()
    frequency = ["Never", "Rarely", "Sometimes", "Frequently"]
    helpful_for_studying = x_train["Helpful for studying"].unique()
    daily_usage = ["< 2 hours", "> 6 hours", "2-4 hours", "4-6 hours"]
    performance_impact = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
    attention_span = x_train["Attention span"].unique()
    preprocessor = ColumnTransformer(transformers=[
        ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal),
        ("ordinal", OrdinalEncoder(categories=[age,
                                               gender,
                                               mobile_system,
                                               frequency,
                                               helpful_for_studying,
                                               daily_usage,
                                               performance_impact,
                                               attention_span,
                                               frequency]), ordinal)
    ])

    # build and train model
    cls = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression())
    ])
    cls.fit(x_train, y_train)

    # prediction
    y_predict = cls.predict(x_test)
    print(classification_report(y_test, y_predict))

    # confusion matrix
    cm = confusion_matrix(y_test, y_predict)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()

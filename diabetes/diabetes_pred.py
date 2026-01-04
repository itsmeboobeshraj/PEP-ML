import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')



data = pd.read_csv("diabetes.csv")
data.head()

data.info()

data.describe()

p= data.hist(figsize=(20,20))
plt.show()


X = data.drop("Outcome", axis=1)
y = data["Outcome"]
y=data.Outcome

plt.figure(figsize=(12,10))
p=sns.heatmap(data.corr(), annot= True,cmap='RdYlGn')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
print("Model accuracy on test data:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))



def get_user_input():
    print("\nEnter the following health details:")
    pregnancies = float(input("Number of Pregnancies (0 if male): "))
    glucose = float(input("Glucose level (mg/dL): "))
    blood_pressure = float(input("Blood Pressure (mm Hg): "))
    skin_thickness = float(input("Skin Thickness (mm): "))
    insulin = float(input("Insulin level (IU/mL): "))
    bmi = float(input("BMI (weight in kg / (height in m)^2): "))
    dpf = float(input("Diabetes Pedigree Function (family history score): "))
    age = float(input("Age (years): "))

    user_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]]

    return user_data



while True:
    choice = input("\nDo you want to check diabetes risk? (yes/no): ").strip().lower()
    if choice != "yes":
        print("Exiting the program.")
        break

    user_data = get_user_input()

    user_data_scaled = scaler.transform(user_data)

    prediction = model.predict(user_data_scaled)[0]
    prediction_proba = model.predict_proba(user_data_scaled)[0][1]

    if prediction == 1:
        print(f"\nPrediction: High chance of diabetes (class 1).")
    else:
        print(f"\nPrediction: Low chance of diabetes (class 0).")

    print(f"Estimated probability of having diabetes: {prediction_proba:.2f}")

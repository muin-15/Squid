import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# --- Part 1: Quiz Question Bank ---
question_bank = {
    "easy": [
        "What is 2 + 2?",
        "What is the capital of India?",
        "Which planet is known as the Red Planet?",
        "In Python, how do you write a comment?",
        "What is 10 ÷ 2?",
        "Who is known as the father of the nation of India?",
        "What does the len() function in Python do?",
        "What shape has 3 sides?",
        "What is 5 × 3?",
        "Which gas do humans need to breathe?"
    ],
    "medium": [
        "What is the square root of 144?",
        "Who invented the light bulb?",
        "What is the output of print(2 ** 4) in Python?",
        "What is 25% of 200?",
        "Which continent is the smallest?",
        "What is the median of [3, 7, 9, 15, 20]?",
        "What is the area of a rectangle with length 12 and width 5?",
        "Who painted the Mona Lisa?",
        "In Python, what is the difference between == and = ?",
        "What is the next prime number after 13?"
    ],
    "hard": [
        "What is the derivative of x²?",
        "Who developed the theory of relativity?",
        "What is the time complexity of binary search?",
        "In Python, what does the zip() function do?",
        "What is the chemical symbol for Gold?",
        "What is the LCM of 24 and 36?",
        "What is the capital of Australia?",
        "In machine learning, what does 'overfitting' mean?",
        "What is the largest desert in the world?",
        "What is the cube root of 729?"
    ]
}

# --- Mode selection for quiz ---
print("Select difficulty mode: easy / medium / hard")
mode = input("Enter your choice: ").strip().lower()

if mode in question_bank:
    print(f"\nYou selected {mode.upper()} mode. Here are your questions:\n")
    for i, q in enumerate(question_bank[mode], start=1):
        print(f"{i}. {q}")
else:
    print("Invalid mode selected.")

# --- Part 2: Sample Dataset for Decision Tree ---
data = {
    "Feature1": [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
    "Feature2": [100, 200, 300, 400, 500, 150, 250, 350, 450, 550],
    "Feature3": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
    "Target":   ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "Yes", "No", "Yes"]
}
df = pd.DataFrame(data)

# Encode Feature3
le = LabelEncoder()
df["Feature3_encoded"] = le.fit_transform(df["Feature3"])

# Encode Target
target_le = LabelEncoder()
df["Target"] = target_le.fit_transform(df["Target"])

# Features & Target
FEATURES = ["Feature1", "Feature2", "Feature3_encoded"]
TARGET = "Target"

X = df[FEATURES]
y = df[TARGET]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n--- Decision Tree Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Add Predictions
df["Predicted_Target"] = target_le.inverse_transform(model.predict(X))
print("\nData with Predictions:")
print(df)

# Visualize Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=FEATURES,
    class_names=target_le.classes_,
    filled=True,
    rounded=True
)
plt.title("Decision Tree Visualization")
plt.show()


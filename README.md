# Squid
Basic Machine learning using scikitlearn 
An interactive Quiz Game where the user selects a difficulty mode (Easy / Medium / Hard).
Based on the selected mode, the app displays a set of questions.
A Decision Tree Classifier (from scikit-learn) is trained on demo data to simulate adaptability for future features (e.g., predicting difficulty progression based on performance).
The project is deployed with Streamlit, making it accessible via a live demo link.

#Features
Difficulty Modes: Easy, Medium, Hard — each with 10 unique questions.
Dynamic Question Selection: Questions change based on user’s chosen mode.
Machine Learning Integration: Uses DecisionTreeClassifier to show how user data could be modeled.
Visualization: Decision tree plotted with matplotlib for explainability.
Web App Deployment: Built with Streamlit for easy demo access.

#working
User selects a difficulty mode (Easy / Medium / Hard).
The app fetches corresponding questions from the question bank.
A simple Decision Tree Model is trained on demo data to showcase ML integration.
Predictions and rules are displayed, along with quiz questions.
The app runs in the browser (Streamlit), no installation needed for the user.

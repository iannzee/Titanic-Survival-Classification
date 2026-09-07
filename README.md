🚢 Titanic Survival Prediction Pipeline
📌 Project Overview
This project features an end-to-end supervised machine learning classification pipeline that predicts whether a passenger survived the Titanic disaster based on demographic and ticketing data. It serves as Level 3 of a progressive AI & Data Science portfolio, transitioning from exploratory analysis and regression into binary classification.

By utilizing classical algorithms and rigorous preprocessing, this model analyzes the factors that determined survival rates, bridging the gap between historical events and predictive data science.

🛠️ Tech Stack
Language: Python
Libraries: Pandas, NumPy, Seaborn, Scikit-Learn
Algorithm: Logistic Regression (with Sigmoid probability mapping)
Environment: Jupyter Notebook / VS Code
🧹 Data Cleaning, Imputation & Preprocessing
Real-world data is messy and requires strategic engineering before it can be fed into mathematical models.

Handling High-Loss Columns: Dropped the deck column entirely, as over 77% of its data was missing.
Redundancy & Data Leakage Prevention: Removed duplicate/overlapping columns (alive, class, embark_town, and who) to prevent multicollinearity and cheating (data leakage) during training.
Smart Imputation: Imputed missing values in the age column using the median age of the passengers to preserve valuable rows.
Negligible Row Removal: Dropped the 2 rows with missing embarked values to ensure a clean feature space.
Feature Encoding (One-Hot Encoding): Transformed categorical text columns (sex, embarked) into numerical binary features using pd.get_dummies(drop_first=True) to avoid the "dummy variable trap".
The final preprocessed dataset consists of 889 passenger records with 10 fully numerical features.

📐 Mathematical Model & Intuition
The pipeline uses Logistic Regression to perform binary classification. Instead of predicting a continuous numerical value, the model uses the Sigmoid Function to squash a linear combination of features into a probability score between $0$ and $1$:

$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)}}$$

If $P \ge 0.5$, the model classifies the passenger as Survived (1).
If $P < 0.5$, the model classifies the passenger as Perished (0).
Model Coefficients (Feature Importance)
The trained weights (coefficients) reveal how each feature mathematically impacts the probability of survival:

Feature	Coefficient	Impact on Survival Probability
fare	0.0022	Slightly Positive (Higher fares increased survival chances)
age	-0.0279	Negative (Older passengers had slightly lower survival chances)
embarked_Q	-0.0677	Negative
parch	-0.3329	Negative
embarked_S	-0.3987	Negative
sex_male	-0.6781	Negative
sibsp	-0.6831	Negative
alone	-0.6847	Negative
pclass	-1.0428	Highly Negative (Lower-class status penalized survival rate)
adult_male	-2.2834	Extremely Negative (Highest penalty to survival probability)
Historical Alignment: The heavily negative coefficient for adult_male (-2.2834) mathematically represents the maritime protocol of "Women and children first". Being an adult male was the single largest determinant of non-survival.
Socioeconomic Impact: The negative coefficient for pclass (-1.0428) confirms that as passenger class index increased (moving from 1st to 3rd class), the survival probability fell dramatically.
📊 Model Evaluation
We split the data using an 80/20 train-test ratio (711 training records and 178 hidden testing records). The model was graded on the unseen test set, yielding the following results:

Accuracy Score: 81.46% (Correctly predicting the fate of 145 out of 178 unseen passengers)
Precision Score: ~80% (High precision shows minimal false survival predictions)
Recall Score: ~72% (Strong ability to catch and identify actual survivors)
Confusion Matrix
The $2 \times 2$ grid displays the exact distribution of predictions:

True Negatives (TN): Correctly predicted to have perished.
False Positives (FP): Predicted to survive, but perished (False Alarms).
False Negatives (FN): Predicted to perish, but survived (Missed cases).
True Positives (TP): Correctly predicted to have survived.
📂 Project Structure
titanic-survival-prediction/
│
├── notebooks/
│   └── titanic_classification.ipynb  # Jupyter Notebook containing the ML pipeline
├── README.md                          # Professional project documentation
🚀 How to Run the Project
Clone this repository:
git clone https://github.com/iannzee/titanic-survival-prediction.git
cd Titanic-Survival-Prediction
Install dependencies: Make sure Python and the required libraries are installed:
pip install pandas numpy scikit-learn seaborn matplotlib
Run the Notebook: Launch VS Code, open titanic_classification.ipynb, and run the cells sequentially to watch the pipeline execute in real-time.

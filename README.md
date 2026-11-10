# AI101-Mini-Project-Group-50

A small machine-learning project that explores employee data and uses **Logistic Regression** to predict whether an employee is likely to stay at or leave a company.

This repository contains the dataset, a Jupyter Notebook with the complete workflow, visual analysis and model evaluation.

---

## Project at a Glance

Employee attrition can affect team continuity, hiring costs and an organisation's day-to-day operations. In this project, we study the factors recorded in the IBM HR Analytics dataset and build a binary classification model.

The target variable is **`Attrition`**:

- `0` — the employee stayed.
- `1` — the employee left.

The notebook follows four stages: understanding the dataset, preparing features, training the model and evaluating its predictions.

**Repository:** [AI101-Mini-Project-Group-50](https://github.com/Aniket-09042008/AI101-Mini-Project-Group-50)

## Dataset Details

| Item | Details |
|---|---|
| Dataset | IBM HR Analytics Employee Attrition & Performance |
| Source | [Kaggle dataset page](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| File name | `WA_Fn-UseC_-HR-Employee-Attrition.csv` |
| Number of records | 1,470 employees |
| Number of original columns | 35 |
| Prediction target | `Attrition` |

The data includes employee details such as age, department, business travel, job role, income, overtime, satisfaction ratings and years of work experience.

### Target distribution

| Attrition class | Employees | Share |
|---|---:|---:|
| Stayed (`No`) | 1,233 | 83.88% |
| Left (`Yes`) | 237 | 16.12% |

The classes are not evenly represented. For that reason, the project reports precision, recall and F1-score along with accuracy.

---

## Files in This Project

Place the notebook and CSV in the same project folder:

```text
AI101-Mini-Project-Group-50/
│
├── Employee_Attrition_Logistic_Regression.ipynb
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── requirements.txt
└── README.md
```
---

## Team Work Plan

The notebook is divided into four parts so that each member has a clear area of contribution.

| Part | Member | Main work |
|---|---|---|
| 1 — Data exploration and Documentation lead | Member 1: Aniket / Aniket-09042008 | Load the data, inspect columns and values, study attrition counts and create initial plots. |
| 2 — Data Preprocessing and PPT lead| Member 2: Jaswanth Adapa / jaswanth1316 | Clean the data, encode categorical features, inspect correlation, prepare train/test data and scale features. |
| 3 — Model Training and Execution lead | Member 3: Kaushal Kejariwal / kaushal-kejariwal | Train Logistic Regression, generate predictions and prepare learning/loss curves. |
| 4 — Model Analysis and Video lead | Member 4: Sarthak Nitin Naik / Sakuta020 | Analyse the confusion matrix, classification metrics, ROC curve and precision-recall curve. |

---

## Environment and Libraries

Use Python 3.10 or newer with Jupyter Notebook.

The notebook uses these Python packages:

| Package | Purpose in this project |
|---|---|
| `pandas` | Reading and handling tabular data |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting model and evaluation curves |
| `seaborn` | Statistical plots and confusion-matrix heatmap |
| `scikit-learn` | Data splitting, scaling, Logistic Regression and evaluation metrics |
| `jupyter` | Running the `.ipynb` notebook |

### Install the packages

Open a terminal and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## Run the Notebook

### 1. Download or clone the project

```bash
git clone https://github.com/Aniket-09042008/AI101-Mini-Project-Group-50.git
```

### 2. Open the project directory

```bash
cd AI101-Mini-Project-Group-50
```

### 3. Check the dataset location

Make sure this file is in the same folder as `Employee_Attrition_Logistic_Regression.ipynb`:

```text
WA_Fn-UseC_-HR-Employee-Attrition.csv
```

The dataset can be downloaded from the Kaggle link in the Dataset Details section.

### 4. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 5. Start Jupyter

```bash
jupyter notebook
```

### 6. Execute the project

1. Open `Employee_Attrition_Logistic_Regression.ipynb`.
2. Choose **Kernel → Restart & Run All**.
3. Let the cells run in order.
4. Scroll through the notebook to view the printed tables, metric values and plots.

Running the notebook from the project directory ensures that the relative CSV path is found.

---

## What the Notebook Does

### Part 1 — Exploring employee information

The first part loads the CSV and checks its basic structure. It displays sample records, column types, missing values, duplicate rows and summary statistics. It also examines how many employees stayed and how many left, with plots to make the distribution easier to understand.

### Part 2 — Preparing the data

The next part removes columns that do not provide useful information for prediction, checks missing values and converts the target into numerical form. Categorical input columns are encoded using `pandas.get_dummies()`.

The notebook also checks correlations between numerical features and removes `YearsInCurrentRole` to reduce overlapping tenure-related information. The data is then divided into training and testing sets, and `StandardScaler` is used to scale the features.

### Part 3 — Training Logistic Regression

A Logistic Regression classifier is trained using the prepared training data. The notebook creates:

- A **learning curve** to compare training and testing accuracy as the training set grows.
- A **loss curve** to compare training and testing log loss for different training-set sizes.
- Predictions for the test data and one example employee.

### Part 4 — Checking model performance

The last part evaluates the test predictions using accuracy, a confusion matrix, precision, recall, F1-score and a classification report. It also plots the ROC curve and the precision-recall curve.

---

## Model Results

The following results are recorded in the uploaded notebook for the 80/20 train-test split with `random_state=42`.

| Metric | Result |
|---|---:|
| Accuracy | **86.73%** |
| Precision for employees who left | **65.38%** |
| Recall for employees who left | **36.17%** |
| F1-score for employees who left | **46.58%** |
| ROC-AUC | **0.7985** |
| Average precision (PR) | **0.5735** |

### Confusion matrix

The test-set confusion matrix is:

| Actual / Predicted | Stayed | Left |
|---|---:|---:|
| Stayed | 238 | 9 |
| Left | 30 | 17 |

This means the model correctly classified 238 employees who stayed and 17 employees who left. It also produced 9 false alarms and missed 30 employees who actually left.

### What these numbers tell us

- The overall accuracy is 86.73%, but accuracy alone does not describe how well the model finds employees who leave.
- The recall for the `Left` class is 36.17%. At the model's default threshold, it finds around 36 of every 100 employees in the test set who actually left.
- The ROC-AUC of 0.7985 indicates that the model can separate the two classes better than random ranking.
- The average precision score of 0.5735 gives another view of performance on the less frequent `Left` class.

Since the dataset has many more employees who stayed than employees who left, recall and the precision-recall curve are important when interpreting the results.

---

## Plots and Evaluation Views

All of the following plots are generated in the notebook output cells.

### Training behaviour

**Learning curve** — compares training accuracy with testing accuracy as more training examples are used. In the saved results, testing accuracy stays around 86–87% across the selected training sizes.

**Loss curve** — compares training log loss and testing log loss. The test log loss decreases overall as more training examples are used.

### Test-set evaluation

**Confusion matrix** — shows correct predictions and the two types of classification error.

**ROC curve** — plots the true positive rate against the false positive rate at different probability thresholds. Its AUC score in this run is **0.7985**.

**Precision-recall curve** — shows the relationship between precision and recall for the `Left` class. The average precision score in this run is **0.5735**.

---

## Notes Before Submission

- Keep the CSV file alongside the notebook so the data-loading cell can find it.
- Run the notebook from a fresh kernel using **Restart & Run All** before committing it.
- The saved correlation-table output in the current notebook shows `NaN` entries. Re-run and verify that cell before quoting individual correlation coefficients in a report or presentation.
- The results above belong to the recorded split and model settings. They may change if the data, split, preprocessing or model settings are changed.
- Add the team's real names, GitHub usernames, presentation link and video link before final submission.

## Presentation and Demo

- **PPT:** `Add presentation link`
- **Demo video:** `Add video link (make sure anyone with the link can view it)`

---

## Use of AI Tools

AI Tools Usage
AI tools were used during coding and development to support the project team.
- Understanding Python and machine learning techniques used in the project.
- Clarifying the use of Pandas and Scikit-learn libraries.
- Identifying and fixing coding errors.
- Improving code readability and organising the project workflow.
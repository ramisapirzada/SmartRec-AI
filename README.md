# SmartRec-AI

## AI-Powered Personalized Learning Resource Recommendation System

SmartRec-AI is a machine-learning-based recommendation system that generates personalized learning-resource recommendations according to a learner's interests, difficulty level, preferred format, and learning goal.

The project was developed as part of the DecodeLabs Virtual Internship.

---

## Project Overview

Finding suitable learning resources can be difficult when learners have different interests, experience levels, and learning objectives.

SmartRec-AI addresses this problem by analyzing the relationship between a learner's interests and the available learning resources, then ranking resources according to their relevance and user preferences.

The system uses a content-based recommendation approach combined with preference-aware ranking.

---

## AI Recommendation Approach

The recommendation pipeline consists of the following stages:

### 1. User Profile

The system accepts:

* Learning interests
* Difficulty level
* Preferred learning format
* Learning goal

### 2. Text Representation

Learning-resource information and user interests are represented using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

### 3. Semantic Similarity

**Cosine similarity** is used to measure the similarity between the user's interests and the learning-resource representations.

### 4. Preference Alignment

The recommendation system considers structured preferences including:

* Difficulty
* Format
* Learning goal

### 5. Hybrid Ranking

The relevance and preference signals are combined into a final ranking score.

Resources with stronger overall alignment are ranked higher.

### 6. Personalized Recommendations

The highest-ranked resources are returned to the learner together with match information and recommendation explanations.

---

## System Architecture

```text
User Learning Profile
        │
        ▼
Interest Processing
        │
        ▼
TF-IDF Representation
        │
        ▼
Cosine Similarity
        │
        ▼
Preference Matching
(Difficulty / Format / Goal)
        │
        ▼
Hybrid Recommendation Score
        │
        ▼
Ranked Learning Resources
        │
        ▼
Streamlit Interface
```

---

## Dataset

The project uses a structured learning-resource dataset containing information such as:

* Resource ID
* Title
* Description
* Category
* Skills
* Difficulty
* Format
* Learning Goal

The dataset was validated for missing values before being used by the recommendation system.

---

## Technologies Used

* Python 3.13
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* Seaborn
* Streamlit
* Jupyter Notebook
* VS Code

---

## Project Structure

```text
SmartRec-AI/
│
├── data/
│   ├── learning_resources.csv
│   └── recommendation_evaluation.csv
│
├── notebooks/
│   └── SmartRec-AI development notebook
│
├── src/
│   ├── __init__.py
│   ├── recommender.py
│   └── models/
│       ├── tfidf_vectorizer.joblib
│       ├── item_vectors.joblib
│       └── catalog.joblib
│
├── app.py
├── README.md
└── requirements.txt
```

---

## Running the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SmartRec-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the application

```bash
streamlit run app.py
```

The Streamlit interface will open in the browser.

---

## Example User Profile

```text
Interests:
Python, Machine Learning, Artificial Intelligence

Difficulty:
Intermediate

Format:
Project

Goal:
Projects
```

The system then ranks learning resources according to semantic relevance and the selected preferences.

---

## Evaluation

The project includes recommendation testing across multiple learner profiles to verify that recommendations respond to different interests and preferences.

Evaluation considers:

* Recommendation ranking
* Semantic relevance
* Preference alignment
* Match scores
* Consistency of ranking

---

## Limitations

The current system is primarily content-based and depends on the information available in the learning-resource dataset.

It does not currently use collaborative filtering based on a large population of historical users.

---

## Future Improvements

Possible future improvements include:

* Larger and more diverse learning-resource datasets
* User feedback and rating integration
* Collaborative filtering
* Embedding-based semantic retrieval
* Learning from user interaction history
* Recommendation feedback loops
* More advanced hybrid recommendation models

---

## Conclusion

SmartRec-AI demonstrates how machine-learning techniques can be applied to personalized recommendation.

By combining TF-IDF text representation, cosine similarity, and preference-aware ranking, the system produces recommendations that adapt to a learner's individual interests and learning requirements.

The project provides an end-to-end workflow from dataset preparation and AI recommendation logic to an interactive application.

# 🧠 Resume-to-Job Matching Using NLP

This project uses Natural Language Processing (NLP) techniques to match candidate resumes with job descriptions. It includes a complete pipeline from PDF extraction to similarity-based job matching using transformer models.

---

## 🔍 Project Overview

The pipeline consists of:

1. **PDF Data Extraction**: Extracts key information (skills, education, job role) from resumes in PDF format.
2. **Exploratory Data Analysis (EDA)**: Cleans and analyzes text data to understand resume patterns.
3. **Resume-to-Job Matching**: Uses transformer embeddings and cosine similarity to rank job descriptions against resumes.

---

## 📁 Datasets Used

### 1. Resumes: Kaggle Resume Dataset
- **About**: Contains over 1000 resumes in both text and PDF formats.
- **CSV Contents**:
  - `ID`: Unique identifier used as the PDF filename.
  - `Resume_str`: Resume content in string format.
  - `Category`: Job category (e.g., HR, IT, Finance, Designer, etc.)
- **Folder Structure**: PDFs are organized in category-based folders.

### 2. Job Descriptions: Hugging Face Job Descriptions Dataset
- Extracted job roles and descriptions for similarity matching.

---

## ⚙️ Implementation Details

### 1. Resume Extraction
- **Libraries Used**: `PyPDF2`, `pdfplumber`, `spaCy`, `re`
- **Extracted Features**:
  - Job Category
  - Key Skills
  - Educational Details (Degrees, Institutions)
- Output is stored in `skills_education.csv`.

### 2. Exploratory Data Analysis (EDA)
- **Libraries**: `pandas`, `seaborn`, `matplotlib`, `contractions`
- **Preprocessing Steps**:
  - Expand contractions (e.g., “can’t” → “cannot”)
  - Remove special characters and stopwords
- **Analysis Performed**:
  - Word frequency distributions
  - Resume length insights

### 3. Resume-Job Matching
- **Libraries**: `transformers`, `spaCy`, `OpenAI API`, `scikit-learn`
- **Method**:
  - Tokenize job descriptions
  - Generate embeddings using `DistilBERT`
  - Compute cosine similarity between resume and job vectors
  - Rank job roles by similarity score

---

## 📊 Evaluation Criteria

- **Functionality**: Effectiveness of resume-job matching
- **Code Quality**: Modular, readable, and well-documented code
- **Generative AI Usage**: Integration of large language models
- **Creativity**: Innovative techniques or workflows
- **Scalability**: Ability to scale to larger datasets

---

## ▶️ How to Run

### ✅ Prerequisites

Install the required libraries:

```bash
pip install pandas numpy pdfplumber PyPDF2 spacy transformers openai sklearn seaborn matplotlib datasets
```

### 🚀 Execution Steps

```bash
# Step 1: Extract resume data
python extract_resume.py

# Step 2: Perform EDA
python eda.py

# Step 3: Match resumes to job descriptions
python match_resumes.py
```

---

## 🚀 Future Improvements

- Fine-tune transformer models on custom resume/job datasets.
- Enhance skill and education entity extraction.
- Build a front-end interface using **Streamlit** or **Gradio**.
- Integrate more job boards and real-time scraping.

---

## 🛠️ Execution Steps

1. **Clone the Repository**:
```bash
git clone https://github.com/yourusername/resume-matcher-nlp.git
cd resume-matcher-nlp
```

2. **Install Required Libraries**:
```bash
pip install pandas numpy pdfplumber PyPDF2 spacy transformers openai sklearn seaborn matplotlib datasets
```

3. **Download Required NLP Models**:
```bash
python -m spacy download en_core_web_sm
```

4. **Run the Pipeline**:
```bash
# Step 1: Extract resume data
python extract_resume.py

# Step 2: Perform EDA
python eda.py

# Step 3: Match resumes to job descriptions
python match_resumes.py
```

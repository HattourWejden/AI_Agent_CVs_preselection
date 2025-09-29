Agent IA de Pré-sélection de CVs
================================

**A Streamlit-powered AI tool for HR professionals to efficiently preselect candidates by analyzing resumes against job descriptions.**

📑 Overview
-----------

This application leverages machine learning and natural language processing to streamline the resume screening process. It analyzes PDF resumes, extracts candidate information, and ranks candidates based on their fit with a provided job description. The tool provides explainable AI insights using LIME and offers a modern, user-friendly interface with interactive features.

✨ Features
----------

*   **Resume Analysis**: Upload multiple PDF resumes for automated processing.
    
*   **Information Extraction**: Extracts candidate details (name, email, phone) using SpaCy.
    
*   **Fit & Similarity Scoring**:
    
    *   Fit Score: Evaluates candidate suitability using a Logistic Regression model with TF-IDF vectorization.
        
    *   Similarity Score: Measures cosine similarity between resumes and job descriptions.
        
*   **Explainable AI**: Provides LIME explanations to highlight key factors in candidate scoring.
    
*   **Interactive UI**:
    
    *   Toggle between card and table views for results.
        
    *   Filter candidates by fit score.
        
    *   Export results as CSV.

🛠️ Requirements
----------------

To run this project, ensure you have the following:

*   **Python**: Version 3.8 or higher
    
*   streamlitpandasPyPDF2scikit-learnlimedatasetsspacynumpy
    
*   python -m spacy download en\_core\_web\_lg
    

🚀 Installation
---------------

Follow these steps to set up the project locally:

1.  git clone https://github.com/HattourWejden/AI_Agent_CVs_preselection.git cd Agent-IA
    

📖 Usage
--------

1.  streamlit run app.py
    
2.  **Access the App**: Open your browser and go to http://localhost:8501.
    
3.  **Interact with the Interface**:
    
    *   **Sidebar**: Enter a job description or use the sample description button.
        
    *   **Main Panel**: Upload one or more PDF resumes.
        
    *   **Results**: View ranked candidates, filter by fit score, and explore LIME explanations.
    
        
4.  **Tips**:
    
    *   Ensure job descriptions are detailed (minimum 50 characters) for accurate scoring.
        
    *   Use the reset button to clear inputs and start over.
        
    
🙏 Acknowledgments
------------------

*   **Streamlit**: For the intuitive web interface framework.
    
*   **SpaCy**: For robust NLP capabilities.
    
*   **LIME**: For explainable AI functionality.
    
*   **Dataset**: [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) for training data.
    


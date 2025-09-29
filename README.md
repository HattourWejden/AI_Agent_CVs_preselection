Agent IA de Pré-sélection de CVs
This is a Streamlit-based web application designed to assist HR professionals in preselecting candidates by analyzing resumes (PDF format) against a job description. The application uses machine learning to evaluate the fit of candidates, extract key information (name, email, phone), and provide explainable AI insights using LIME.
Features

Resume Upload: Upload multiple PDF resumes for analysis.
Job Description Input: Enter a job description to evaluate candidate suitability.
Candidate Information Extraction: Automatically extracts name, email, and phone number from resumes using SpaCy.
Fit Scoring: Uses a Logistic Regression model with TF-IDF vectorization to calculate a fit score for each candidate.
Similarity Scoring: Computes cosine similarity between resumes and the job description.
Explainable AI: Provides LIME explanations to highlight key factors influencing candidate fit scores.
Interactive Interface: Features a modern UI with card-based or table-based views, a fit score filter, and CSV export functionality.
Progress Feedback: Includes a progress bar for processing multiple resumes and clear user feedback messages.

Screenshots
Screenshot 1: Main Interface
Description: The main interface showing the resume upload section, job description input in the sidebar, and candidate analysis results in card view.
Screenshot 2: LIME Explanation
Description: An expanded candidate card displaying the LIME explanation, highlighting key terms contributing to the fit score.
Prerequisites

Python 3.8 or higher

Required Python packages (listed in requirements.txt):
streamlit
pandas
PyPDF2
scikit-learn
lime
datasets
spacy
numpy


SpaCy language model:
python -m spacy download en_core_web_lg



Installation

Clone the repository:
git clone https://github.com/your-username/cv-preselection-agent.git
cd cv-preselection-agent


Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required packages:
pip install -r requirements.txt


Install the SpaCy language model:
python -m spacy download en_core_web_lg


Add a logo.png file to the project root for the sidebar logo (optional).


Usage

Run the Streamlit application:
streamlit run app.py


Open your browser and navigate to http://localhost:8501.

In the sidebar:

Enter a job description or click "Utiliser un exemple de description" for a sample.
Upload one or more PDF resumes in the main interface.


View the results:

Candidates are ranked by fit score in either card or table view.
Filter candidates by minimum fit score using the slider.
Click "Voir l'explication" to see LIME explanations for a candidate's score.
Download the results as a CSV file using the "Télécharger les résultats (CSV)" button.



Project Structure
cv-preselection-agent/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── logo.png            # Sidebar logo image
├── screenshots/        # Folder for screenshots
│   ├── main_interface.png
│   ├── lime_explanation.png
└── README.md           # This file

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built with Streamlit for the web interface.
Uses SpaCy for natural language processing.
Leverages LIME for explainable AI.

Dataset provided by cnamuangtoun/resume-job-description-fit.

# Job recommender via natural language processing
## 1. Overview
The project is done during the last three weeks of Data Science Bootcamp at WBS Coding School. The idea of project is to help job seekers to find right jobs based on their CV.

Three popular platforms for job search is used for collecting data:

- [Linkedin](https://www.linkedin.com/)
- [Glassdoor](https://www.glassdoor.de/)
- [Indeed](https://de.indeed.com/)

For my purpose, I only focus on jobs which are related to Data.

Tech stacks: **selenium, deep translator, text extraction, spacy, natural language processing, recommender system**.
## 2. Data Aquisition
I use Selenium to dynamically collect data from 3 different platforms.

- [Linkedin](https://www.linkedin.com/)
- [Glassdoor](https://www.glassdoor.de/)
- [Indeed](https://de.indeed.com/)

The market is Germany and two keywords are used **Data Scientist** and **Data Analyst**.

The data is collected every week to update all new job posts. Overall, there are about 3000 job posts collected each week.
## 3. Natural Language Processing
Some libraries in natural language processing are used to extract information from CV and job description.

- First, language detection is used to get language from job description. Then we use **deep translator** to translate all job description to english.

- When a CV is uploaded to the app. The system will use  **skillNer** and **spacy** model to extract both technical and soft skills. All skills from CV will be matched to job descriptions and finally recommend top 20 jobs.

## 4. Deploy model

I use streamlit to depploy the app.
[App Job recommender](https://thaingoc273-job-recommender-via--job-recommender-via-nlp-xatvqc.streamlitapp.com/)

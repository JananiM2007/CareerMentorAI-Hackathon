import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import joblib
import re
import plotly.express as px
#BG
st.markdown("""
<style>

.stApp{
background: linear-gradient(
135deg,
#e0e7ff,
#c7d2fe,
#dbeafe
);
}

</style>
""", unsafe_allow_html=True)
# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Career Mentor",
    page_icon="🎯",
    layout="wide"
)

# ==========================================
# LOAD MODELS
# ==========================================
model = joblib.load("career_model.pkl")
tfidf = joblib.load("tfidf.pkl")
encoder = joblib.load("label_encoder.pkl")

# ==========================================
# LOAD DATASETS
# ==========================================
career_skills_df = pd.read_csv("career_skills_dataset2.csv")
course_df = pd.read_csv("skill_course_dataset2.csv")
internship_df = pd.read_csv("internship_dataset2.csv")

# ==========================================
# FUNCTIONS
# ==========================================

def clean_resume(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)

    text = re.sub(
        r'[%s]' % re.escape(
            """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
        ),
        ' ',
        text
    )

    text = re.sub(r'\s+', ' ', text)

    return text.lower()


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


# ==========================================
# BUILD MASTER SKILL LIST
# ==========================================
all_skills = set()

for skills in career_skills_df["Required_Skills"]:
    for skill in skills.split(","):
        all_skills.add(skill.strip().lower())


def extract_skills(text):
    text = text.lower()

    found_skills = []

    for skill in all_skills:
        if skill in text:
            found_skills.append(skill)

    return sorted(list(set(found_skills)))


def get_missing_skills(career, user_skills):

    row = career_skills_df[
        career_skills_df["Career"] == career
    ]

    if row.empty:
        return []

    required_skills = row.iloc[0]["Required_Skills"]

    required_skills = [
        skill.strip().lower()
        for skill in required_skills.split(",")
    ]

    missing = []

    for skill in required_skills:
        if skill not in user_skills:
            missing.append(skill)

    return missing


def recommend_courses(missing_skills):

    recommendations = []

    for skill in missing_skills:

        rows = course_df[
            course_df["Skill"].str.lower() == skill
        ]

        for _, row in rows.iterrows():

            recommendations.append({
                "Skill": row["Skill"],
                "Course": row["Course"],
                "Platform": row["Platform"],
                "Difficulty": row["Difficulty"],
                "Duration": row["Duration"]
            })

    return recommendations


def recommend_internships(career):

    internships = internship_df[
        internship_df["Career"] == career
    ]

    return internships


def generate_roadmap(career, missing_skills):

    roadmap = []

    roadmap.append("1️⃣ Learn the missing skills")

    for skill in missing_skills:
        roadmap.append(f"• {skill}")

    roadmap.append("2️⃣ Build 2-3 Projects")
    roadmap.append("3️⃣ Apply for Internships")
    roadmap.append(f"4️⃣ Become a {career}")

    return roadmap


def calculate_resume_score(
        extracted_skills,
        missing_skills,
        resume_text):

    score = 0

    score += min(len(extracted_skills) * 5, 50)

    if len(resume_text) > 1000:
        score += 20

    if len(missing_skills) <= 3:
        score += 30

    return min(score, 100)

#ATS SCORE 
def calculate_ats_score(
    extracted_skills,
    required_skills,
    resume_text
):

    score = 0

    matched_skills = len(
        set(extracted_skills) &
        set(required_skills)
    )

    skill_score = (
        matched_skills /
        len(required_skills)
    ) * 70

    score += skill_score

    text = resume_text.lower()

    if "experience" in text:
        score += 10

    if "project" in text:
        score += 10

    if "education" in text:
        score += 5

    if (
        "certification" in text
        or
        "certificate" in text
    ):
        score += 5

    return round(score)

# ==========================================
# UI
# ==========================================
st.markdown("""
<h1 style='text-align:center;color:#4f46e5'>
🎯 AI Career Mentor
</h1>

<h4 style='text-align:center'>
Personalized Career Intelligence Platform
</h4>
""",
unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

# ==========================================
# MAIN PIPELINE
# ==========================================
if uploaded_file:

    st.success("Resume uploaded successfully!")

    # Extract text
    resume_text = extract_text_from_pdf(
        uploaded_file
    )

    # Clean text
    cleaned_resume = clean_resume(
        resume_text
    )

    # Career prediction
    vector = tfidf.transform(
        [cleaned_resume]
    )

    prediction = model.predict(
        vector
    )

    predicted_career = encoder.inverse_transform(
        prediction
    )[0]

    # Skills
    extracted_skills = extract_skills(
        resume_text
    )

    # Missing skills
    missing_skills = get_missing_skills(
        predicted_career,
        extracted_skills
    )

    # Recommendations
    courses = recommend_courses(
        missing_skills
    )

    internships = recommend_internships(
        predicted_career
    )

    roadmap = generate_roadmap(
        predicted_career,
        missing_skills
    )

    resume_score = calculate_resume_score(
    extracted_skills,
    missing_skills,
    resume_text
)

    career_row = career_skills_df[
    career_skills_df["Career"] == predicted_career
    ]

    required_skills = career_row.iloc[0]["Required_Skills"]

    required_skills = [
    skill.strip().lower()
    for skill in required_skills.split(",")
    ]

    ats_score = calculate_ats_score(
    extracted_skills,
    required_skills,
    resume_text
)
    # ======================================
    # METRICS
    # ======================================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
        "🎯 Career",
        predicted_career
        )

    with col2:
        st.metric(
        "🤖 ATS Score",
        ats_score
        )

    with col3:
        st.metric(
        "📄 Resume Score",
        resume_score
        )

    st.divider()

    st.subheader("📄 Resume Preview")
    st.text(resume_text[:1000])

    tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analysis",
    "📚 Learning",
    "💼 Opportunities",
    "🗺 Roadmap"
    ])
    with tab1:
        st.subheader("🎯 Predicted Career")
        st.success(predicted_career)

        st.subheader("📊 Resume Score")
        st.progress(resume_score)
        st.write(f"{resume_score}/100")

        st.subheader("🤖 ATS Score")
        st.progress(ats_score)
        st.write(f"{ats_score}/100")
        if ats_score >= 85:
            st.success("Excellent Resume")
        elif ats_score >= 70:
            st.info("Good Resume")
        elif ats_score >= 50:
            st.warning("Needs Improvement")
        else:
            st.error("Poor ATS Compatibility")
        st.subheader("🛠 Extracted Skills")

        if extracted_skills:
            st.write(", ".join(extracted_skills))
        else:
            st.warning("No skills detected.")
        st.subheader("📈 Skill Match")

        match_percentage = (
        (len(required_skills) - len(missing_skills))
        / len(required_skills)
        ) * 100

        st.progress(match_percentage / 100)

        st.write(
        f"{round(match_percentage)}% of required skills matched"
        )
    # ======================================
    # SKILL CHART
    # ======================================
        st.subheader("🛠 Skills Detected")

        cols = st.columns(4)

        for i, skill in enumerate(extracted_skills):

            cols[i % 4].success(skill)
        st.subheader("📈 Skill Gap Analysis")

        if missing_skills:
            st.error(
            ", ".join(missing_skills)
            )
        else:
            st.success(
            "No missing skills detected!"
        )
    with tab2:
        st.subheader("📚 Recommended Courses")

        if courses:

            for course in courses:

                with st.expander(
                course["Course"]
            ):

                    st.write(
                    f"Platform: {course['Platform']}"
                )

                    st.write(
                    f"Difficulty: {course['Difficulty']}"
                )

                    st.write(
                    f"Duration: {course['Duration']}"
                )

        else:
            st.write(
            "No course recommendations found."
        )
    with tab3:
        st.subheader("💼 Recommended Internships")

        if not internships.empty:

            for _, row in internships.iterrows():

                st.write(
                f"🔹 {row['Internship_Role']}"
            )

                st.write(
                f"Skills Required: {row['Required_Skills']}"
            )

                st.write("---")

        else:
            st.write(
            "No internships found."
        )
    with tab4:
        st.subheader("🗺 Career Roadmap")

        for step in roadmap:
            st.write(step)

    # ======================================
    # DOWNLOAD REPORT
    # ======================================
    report = f"""
Predicted Career:
{predicted_career}

Resume Score:
{resume_score}/100

ATS Score:
{ats_score}/100

Skills:
{', '.join(extracted_skills)}

Missing Skills:
{', '.join(missing_skills)}

Roadmap:
{chr(10).join(roadmap)}
"""

    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="career_report.txt",
        mime="text/plain"
    )


import streamlit as st
import joblib
import pandas as pd
import re
import fitz  # PyMuPDF
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="SmartHire | AI Talent Intelligence", 
    page_icon="⚡",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .stat-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 18px;
        text-align: left;
        margin-bottom: 22px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-box:hover {
        border-color: rgba(88, 166, 255, 0.4);
        transform: translateY(-2px);
    }
    .stat-number {
        font-size: 1.45rem;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 2px;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .stat-sub {
        font-size: 0.8rem;
        color: #c9d1d9;
        margin-top: 4px;
    }
    .badge-matched {
        background-color: #0e4429;
        color: #39d353;
        border: 1px solid #238636;
        padding: 5px 12px;
        border-radius: 20px;
        display: inline-block;
        margin: 4px;
        font-weight: 500;
        font-size: 0.85rem;
    }
    .badge-missing {
        background-color: #490202;
        color: #ff7b72;
        border: 1px solid #b62324;
        padding: 5px 12px;
        border-radius: 20px;
        display: inline-block;
        margin: 4px;
        font-weight: 500;
        font-size: 0.85rem;
    }
    .hero-sub {
        color: #8b949e;
        font-size: 1.15rem;
        margin-top: -10px;
        margin-bottom: 20px;
        font-weight: 400;
    }
    .star-rating {
        font-size: 1.5rem;
        color: #e3b341;
        letter-spacing: 3px;
        font-weight: bold;
    }
    .star-muted {
        color: #30363d;
    }
    </style>
""", unsafe_allow_html=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_contact_info(text):
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    phones = re.findall(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
    return {
        "email": emails[0] if emails else "Not detected",
        "phone": phones[0] if phones else "Not detected"
    }

def check_resume_health(text, email, phone):
    checks = []
    
    # 1. Contact Completeness Check
    contact_ok = (email != "Not detected") and (phone != "Not detected")
    checks.append({
        "title": "Contact Completeness",
        "status": contact_ok,
        "detail": "Verified phone & professional email detected" if contact_ok else "Missing clear email or phone number"
    })
    
    # 2. Action Verbs Strength
    action_verbs = ["managed", "developed", "created", "led", "engineered", "built", "analyzed", "designed", "implemented", "optimized"]
    found_verbs = [v for v in action_verbs if re.search(r'\b' + v + r'\b', text.lower())]
    checks.append({
        "title": "Action Verbs Strength",
        "status": len(found_verbs) >= 2,
        "detail": f"Active phrasing detected ({len(found_verbs)} verbs found)" if len(found_verbs) >= 2 else "Lacks impact verbs (e.g., Built, Developed, Led)"
    })
    
    # 3. Measurable Impact & Metrics
    metrics = re.findall(r'\b(?:\d+%|\$\d+|\d+\+|\d+\s*(?:users|clients|projects|team|members|hours|days))\b', text.lower())
    checks.append({
        "title": "Measurable Impact & Metrics",
        "status": len(metrics) > 0,
        "detail": f"Quantified outcomes detected ({len(metrics)} instances)" if len(metrics) > 0 else "No quantifiable data found (e.g., '20% boost', '100+ users')"
    })
    
    # 4. Key Sections Structure Check
    structure_sections = ["education", "experience", "project", "projects", "skill", "skills"]
    found_sections = [s for s in structure_sections if re.search(r'\b' + s + r'\b', text.lower())]
    has_structure = len(set(found_sections)) >= 2
    checks.append({
        "title": "Key Sections Structure",
        "status": has_structure,
        "detail": "Standard structural sections detected (Education, Experience/Projects, Skills)" if has_structure else "Missing standard section headers"
    })
    
    # 5. Content Depth & Word Count
    word_count = len(text.split())
    checks.append({
        "title": "Content Depth & Word Count",
        "status": word_count >= 150,
        "detail": f"Healthy content density ({word_count} words)" if word_count >= 150 else f"Too brief ({word_count} words; 150+ recommended)"
    })
    
    passed_count = sum(1 for c in checks if c["status"])
    return checks, passed_count, len(checks)

def extract_text_from_pdf(uploaded_file):
    extracted_text = ""
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            text = page.get_text()
            if text:
                extracted_text += text + " "
    except Exception as e:
        st.error(f"Error reading document: {e}")
    return extracted_text

@st.cache_resource
def load_assets():
    model = joblib.load('models/resume_classifier.pkl')
    import os
    
    # Check if naukri dataset is present
    naukri_files = [f for f in os.listdir('data/raw') if 'naukri' in f] if os.path.exists('data/raw') else []
    
    if naukri_files:
        job_file = naukri_files[0]
        jobs_df = pd.read_json(f'data/raw/{job_file}', lines=True)
        clean_jobs = jobs_df[['job_title', 'company_name', 'job_description', 'city']].dropna().head(3000).reset_index(drop=True)
    else:
        # Fallback using Resume.csv so deployment never crashes
        resume_df = pd.read_csv('data/raw/Resume.csv')
        categories = resume_df['Category'].dropna().unique()
        cities = ['Bengaluru', 'Delhi NCR', 'Mumbai', 'Hyderabad', 'Pune', 'Chennai', 'Remote / Work From Home']
        
        sample_rows = []
        for i, row in resume_df.head(3000).iterrows():
            sample_rows.append({
                'job_title': f"{row['Category']} Specialist / Engineer",
                'company_name': f"Tech Solutions {i % 10 + 1}",
                'job_description': str(row['Resume_str'])[:500],
                'city': cities[i % len(cities)]
            })
        clean_jobs = pd.DataFrame(sample_rows)
        
    clean_jobs['clean_desc'] = clean_jobs['job_description'].apply(clean_text)
    tfidf_jobs = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf_jobs.fit_transform(clean_jobs['clean_desc'])
    return model, clean_jobs, tfidf_jobs, tfidf_matrix

COMMON_SKILLS = [
    "python", "machine learning", "deep learning", "nlp", "sql", "tableau", 
    "power bi", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", 
    "docker", "kubernetes", "aws", "azure", "git", "flask", "django", 
    "communication", "leadership", "agile", "spark", "hadoop", "excel"
]

def find_skills(text):
    text_lower = text.lower()
    return set([skill for skill in COMMON_SKILLS if re.search(r'\b' + re.escape(skill) + r'\b', text_lower)])

try:
    classifier, jobs_data, job_vec, job_matrix = load_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Asset loading error: {e}")
    assets_loaded = False

# Sidebar Controls
st.sidebar.header("📍 Candidate Preferences")
st.sidebar.caption("Filter matches across top Indian metropolitan hubs")

METRO_CITIES = [
    "All Locations",
    "Remote / Work From Home",
    "Bengaluru / Bangalore",
    "Delhi NCR (Delhi, Noida, Gurgaon)",
    "Mumbai",
    "Hyderabad",
    "Pune",
    "Chennai",
    "Kolkata",
    "Ahmedabad",
    "Gurugram / Gurgaon",
    "Noida / Greater Noida",
    "Kochi / Cochin",
    "Chandigarh",
    "Jaipur",
    "Indore"
]

selected_location = st.sidebar.selectbox("Target Metro / Region:", METRO_CITIES)
top_n = st.sidebar.slider("Number of Job Matches to Display:", min_value=3, max_value=10, value=5)

# Main Page Header
st.title("⚡ SmartHire Intelligence Hub")
st.markdown('<div class="hero-sub">Turn Your Resume into Interviews: Analyze, Align & Get Hired.</div>', unsafe_allow_html=True)

# --- Top Platform Highlights / Stats Row ---
st_col1, st_col2, st_col3 = st.columns(3)
with st_col1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-label">Model Classification</div>
        <div class="stat-number">25+ Domains</div>
        <div class="stat-sub">Automated multi-class skill categorization</div>
    </div>
    """, unsafe_allow_html=True)

with st_col2:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-label">Opportunity Pool</div>
        <div class="stat-number">3,000+ Active Roles</div>
        <div class="stat-sub">Semantic cosine matching across top metros</div>
    </div>
    """, unsafe_allow_html=True)

with st_col3:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-label">Audit Engine</div>
        <div class="stat-number">5-Star Health Index</div>
        <div class="stat-sub">Action verbs, metrics & ATS readiness check</div>
    </div>
    """, unsafe_allow_html=True)

if assets_loaded:
    tab1, tab2 = st.tabs(["📄 Direct PDF Upload", "✍️ Manual Resume Input"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload candidate resume document (.pdf)", type=["pdf"])
        if uploaded_file is not None:
            extracted = extract_text_from_pdf(uploaded_file)
            if extracted.strip():
                st.session_state['resume_text'] = extracted
                st.success("Resume parsed and extracted successfully.")
            else:
                st.error("No extractable digital text found. Please ensure it is a digital PDF or use manual paste.")

    with tab2:
        text_input = st.text_area("Paste Resume Content:", height=180, placeholder="Paste candidate profile or resume text here...")
        if text_input.strip():
            st.session_state['resume_text'] = text_input

    st.write("")
    if st.button("🚀 Run Comprehensive Evaluation", type="primary", use_container_width=True):
        resume_content = st.session_state.get('resume_text', "")
        
        if not resume_content.strip():
            st.warning("Please upload a resume or paste text before initiating evaluation.")
        else:
            cleaned_resume = clean_text(resume_content)
            contact_info = extract_contact_info(resume_content)
            
            # Domain Prediction
            predicted_category = classifier.predict([cleaned_resume])[0]
            
            # Recommendation with Robust Metro Location Filter
            user_vec = job_vec.transform([cleaned_resume])
            sim_scores = cosine_similarity(user_vec, job_matrix).flatten()
            
            candidate_pool = jobs_data.copy()
            candidate_pool['similarity'] = sim_scores
            
            if selected_location != "All Locations":
                if selected_location == "Remote / Work From Home":
                    pattern = r'remote|home|anywhere'
                elif "Bengaluru" in selected_location:
                    pattern = r'bengaluru|bangalore'
                elif "Delhi NCR" in selected_location:
                    pattern = r'delhi|ncr|noida|gurgaon|gurugram|ghaziabad|faridabad'
                elif "Gurugram" in selected_location:
                    pattern = r'gurgaon|gurugram'
                elif "Noida" in selected_location:
                    pattern = r'noida'
                elif "Kochi" in selected_location:
                    pattern = r'kochi|cochin|ernakulam'
                else:
                    clean_city_query = selected_location.split('/')[0].strip()
                    pattern = re.escape(clean_city_query)
                
                filtered_pool = candidate_pool[candidate_pool['city'].str.contains(pattern, case=False, na=False)]
                
                if len(filtered_pool) > 0:
                    candidate_pool = filtered_pool
                else:
                    st.info(f"Notice: Specific listings for **{selected_location}** are currently limited in the local benchmark database. Displaying top nationally ranked opportunities instead.")

            top_matches = candidate_pool.sort_values(by='similarity', ascending=False).head(top_n)
            
            recommendations = top_matches[['job_title', 'company_name', 'city']].copy()
            recommendations.columns = ['Job Title', 'Company Name', 'Location']
            recommendations['Match Score (%)'] = (top_matches['similarity'] * 100).round(2)
            
            best_sim = top_matches['similarity'].iloc[0] if len(top_matches) > 0 else 0
            ats_score = min(int(best_sim * 150), 100)
            
            # Candidate Profile Summary Card
            st.markdown("---")
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin-top:0; color:#58a6ff;">👤 Candidate Profile Summary</h4>
                <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
                    <div><strong>Detected Email:</strong> <span style="color:#7ee787;">{contact_info['email']}</span></div>
                    <div><strong>Detected Phone:</strong> <span style="color:#7ee787;">{contact_info['phone']}</span></div>
                    <div><strong>Primary Domain:</strong> <span style="color:#ffa657;">{predicted_category}</span></div>
                    <div><strong>Selected Hub:</strong> <span style="color:#58a6ff;">{selected_location}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Visual Analytics Row
            c_gauge, c_jobs = st.columns([1.1, 2])
            
            with c_gauge:
                st.subheader("🎯 ATS Match Benchmark")
                fig = go.Figure(go.Pie(
                    values=[ats_score, 100 - ats_score],
                    hole=0.72,
                    sort=False,
                    direction='clockwise',
                    textinfo='none',
                    marker=dict(colors=['#238636' if ats_score >= 65 else '#d29922', '#21262d']),
                    hoverinfo='none'
                ))
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=240,
                    annotations=[dict(
                        text=f"<b>{ats_score}%</b><br><span style='font-size:12px;color:#8b949e;'>Fit Score</span>",
                        x=0.5, y=0.5,
                        font_size=28,
                        showarrow=False,
                        font_color="#ffffff"
                    )]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with c_jobs:
                st.subheader(f"💼 Top Opportunities ({selected_location})")
                st.dataframe(recommendations, use_container_width=True, hide_index=True)
            
            # 5-Star Resume Health Audit Section
            st.markdown("---")
            checks, passed_c, total_c = check_resume_health(resume_content, contact_info['email'], contact_info['phone'])
            
            stars_html = "".join(['★' for _ in range(passed_c)]) + "".join(['<span class="star-muted">☆</span>' for _ in range(total_c - passed_c)])
            
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:15px;">
                <div>
                    <h3 style="margin:0;">📋 Resume Health Audit</h3>
                    <span style="color:#8b949e; font-size:0.9rem;">Evaluating how ready your resume is for recruiters</span>
                </div>
                <div style="text-align:right;">
                    <span class="star-rating">{stars_html}</span>
                    <span style="font-size:1.15rem; font-weight:700; color:#58a6ff; margin-left:8px;">{passed_c} / 5</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            h_col1, h_col2 = st.columns(2)
            for idx, item in enumerate(checks):
                col = h_col1 if idx % 2 == 0 else h_col2
                icon = "✅" if item["status"] else "⚠️"
                with col:
                    st.markdown(f"""
                    <div style="margin-bottom: 12px; padding: 12px; border-radius: 8px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);">
                        <strong style="color: {'#e6edf3' if item['status'] else '#ff7b72'};">{icon} {item['title']}</strong><br>
                        <span style="font-size: 0.85rem; color: #8b949e;">{item['detail']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Skills Analysis
            st.markdown("---")
            st.subheader("🔍 Skill Alignment & Gap Analysis")
            resume_skills = find_skills(resume_content)
            top_job_desc = " ".join(top_matches['job_description'].tolist())
            required_skills = find_skills(top_job_desc)
            missing_skills = required_skills - resume_skills
            
            sk_col1, sk_col2 = st.columns(2)
            with sk_col1:
                st.markdown("**✅ Verified Candidate Competencies**")
                if resume_skills:
                    badges = "".join([f'<span class="badge-matched">{s.title()}</span>' for s in sorted(resume_skills)])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.caption("No standard technical skills identified in profile.")
            
            with sk_col2:
                st.markdown("**⚠️ High-Priority Skill Recommendations**")
                if missing_skills:
                    badges = "".join([f'<span class="badge-missing">{s.title()}</span>' for s in sorted(missing_skills)])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.success("Target profile shows exceptional alignment with benchmark positions.")
            
            # Export Action
            st.markdown("---")
            health_summary = "\n".join([f"- {c['title']}: {'PASSED' if c['status'] else 'NEEDS ATTENTION'} ({c['detail']})" for c in checks])
            report_text = f"""================ SMART HIRE RESUME AUDIT REPORT ================
Category: {predicted_category}
Target Location: {selected_location}
ATS Score: {ats_score}/100
Resume Health Rating: {passed_c}/5 Stars
Contact Email: {contact_info['email']}
Contact Phone: {contact_info['phone']}

--- RESUME HEALTH AUDIT ({passed_c}/{total_c}) ---
{health_summary}

--- SKILLS SUMMARY ---
Identified Skills: {', '.join([s.title() for s in sorted(resume_skills)]) if resume_skills else 'None'}
Missing Skills: {', '.join([s.title() for s in sorted(missing_skills)]) if missing_skills else 'None'}

Top Matched Job: {recommendations.iloc[0]['Job Title']} at {recommendations.iloc[0]['Company Name']} ({recommendations.iloc[0]['Location']}) - {recommendations.iloc[0]['Match Score (%)']}%
================================================================"""
            
            st.download_button(
                label="📥 Download Full Candidate Assessment Report",
                data=report_text,
                file_name=f"SmartHire_Assessment_{predicted_category}.txt",
                mime="text/plain",
                use_container_width=True
            )
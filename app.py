import streamlit as st
from pathlib import Path
import sys

# ---------------------------------------------------------
# PROJECT SETUP
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender import RecommendationEngine


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="SmartRec-AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.15rem;
        opacity: 0.75;
        margin-bottom: 2rem;
    }

    .recommendation-card {
        padding: 1.3rem;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1rem;
    }

    .score {
        font-size: 1.4rem;
        font-weight: 700;
    }

    .tag {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        margin-right: 0.3rem;
        font-size: 0.8rem;
        border: 1px solid rgba(128,128,128,0.3);
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOAD AI ENGINE
# ---------------------------------------------------------

@st.cache_resource
def load_engine():

    model_dir = PROJECT_ROOT / "src" / "models"

    return RecommendationEngine(
        model_dir=model_dir
    )


try:
    engine = load_engine()
except Exception as e:

    st.error(
        "SmartRec-AI could not load the recommendation engine."
    )

    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 SmartRec-AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered personalized learning resource recommendation system
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("🎯 Your Learning Profile")

    interests = st.text_area(
        "What do you want to learn?",
        placeholder=(
            "Example: Python, Machine Learning, "
            "Artificial Intelligence, Computer Vision"
        ),
        height=120
    )

    difficulty = st.selectbox(
        "Difficulty Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

    preferred_format = st.selectbox(
        "Preferred Format",
        [
            "Course",
            "Tutorial",
            "Project",
            "Article",
            "Video",
            "Book"
        ]
    )

    goal = st.selectbox(
        "Learning Goal",
        [
            "Fundamentals",
            "Projects",
            "Career",
            "Interview",
            "Advanced Skills"
        ]
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    recommend_button = st.button(
        "✨ Generate Recommendations",
        use_container_width=True
    )


# ---------------------------------------------------------
# INTRODUCTION
# ---------------------------------------------------------

if not recommend_button:

    st.info(
        "Enter your learning interests and preferences "
        "in the sidebar to generate personalized recommendations."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "AI Method",
            "TF-IDF"
        )

    with col2:
        st.metric(
            "Ranking",
            "Hybrid"
        )

    with col3:
        st.metric(
            "Personalization",
            "Enabled"
        )

    st.markdown(
        '<div class="section-title">How SmartRec-AI Works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        **1. Understand your interests**

        Your learning interests are converted into a numerical
        representation using TF-IDF.

        **2. Measure semantic relevance**

        Cosine similarity compares your interests with the
        learning-resource descriptions.

        **3. Consider your preferences**

        Difficulty, format, and learning goal are incorporated
        into the ranking.

        **4. Generate personalized recommendations**

        Resources are ranked using a hybrid AI scoring approach
        and the highest-ranked resources are presented first.
        """
    )

    st.markdown(
        '<div class="section-title">Why AI?</div>',
        unsafe_allow_html=True
    )

    st.write(
        "SmartRec-AI uses machine-learning-based text "
        "representation and similarity analysis to identify "
        "resources that are relevant to a user's learning profile."
    )


# ---------------------------------------------------------
# RECOMMENDATIONS
# ---------------------------------------------------------

if recommend_button:

    if not interests.strip():

        st.warning(
            "Please enter at least one learning interest."
        )

        st.stop()

    with st.spinner(
        "Analyzing your learning profile..."
    ):

        try:

            recommendations = engine.recommend(
                interests=interests,
                difficulty=difficulty,
                preferred_format=preferred_format,
                goal=goal,
                top_n=top_n
            )

        except Exception as e:

            st.error(
                "An error occurred while generating recommendations."
            )

            st.exception(e)
            st.stop()


    if recommendations.empty:

        st.warning(
            "No recommendations were found. "
            "Try broader learning interests."
        )

        st.stop()


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    st.success(
        f"{len(recommendations)} personalized recommendations generated."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Resources",
            len(recommendations)
        )

    with col2:

        st.metric(
            "Best Match",
            f"{recommendations['match_percentage'].max():.1f}%"
        )

    with col3:

        st.metric(
            "Average Match",
            f"{recommendations['match_percentage'].mean():.1f}%"
        )


    st.markdown(
        '<div class="section-title">🏆 Recommended For You</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # RECOMMENDATION CARDS
    # -----------------------------------------------------

    for rank, (_, row) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        title = row.get(
            "title",
            "Learning Resource"
        )

        description = row.get(
            "description",
            "Personalized learning resource."
        )

        category = row.get(
            "category",
            "Learning"
        )

        resource_difficulty = row.get(
            "difficulty",
            "Not specified"
        )

        resource_format = row.get(
            "format",
            "Not specified"
        )

        resource_goal = row.get(
            "goal",
            "Not specified"
        )

        match = float(
            row.get(
                "match_percentage",
                0
            )
        )

        similarity = float(
            row.get(
                "content_similarity",
                0
            )
        )

        score = float(
            row.get(
                "final_score",
                0
            )
        )

        st.markdown(
            f"""
            <div class="recommendation-card">

            <h3>#{rank} &nbsp; {title}</h3>

            <p>{description}</p>

            <span class="tag">{category}</span>
            <span class="tag">{resource_difficulty}</span>
            <span class="tag">{resource_format}</span>
            <span class="tag">{resource_goal}</span>

            <br><br>

            <div class="score">
            Match Score: {match:.1f}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(
            f"🔎 Why was '{title}' recommended?"
        ):

            st.write(
                f"""
                **Semantic relevance:** {similarity:.3f}

                **Final AI ranking score:** {score:.3f}

                **Difficulty:** {resource_difficulty}

                **Format:** {resource_format}

                **Learning goal:** {resource_goal}

                This resource was ranked using semantic similarity
                between your interests and the resource information,
                combined with your selected learning preferences.
                """
            )


    # -----------------------------------------------------
    # SCORE VISUALIZATION
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Recommendation Scores</div>',
        unsafe_allow_html=True
    )

    chart_data = recommendations[
        [
            "title",
            "match_percentage"
        ]
    ].copy()

    chart_data = chart_data.set_index(
        "title"
    )

    st.bar_chart(
        chart_data
    )


    # -----------------------------------------------------
    # DATA TABLE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📋 Recommendation Details</div>',
        unsafe_allow_html=True
    )

    display_columns = [
        "title",
        "category",
        "difficulty",
        "format",
        "goal",
        "match_percentage"
    ]

    available_columns = [
        col
        for col in display_columns
        if col in recommendations.columns
    ]

    st.dataframe(
        recommendations[available_columns],
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "SmartRec-AI | AI-powered personalized learning "
    "resource recommendation system"
)
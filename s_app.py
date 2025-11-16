import streamlit as st
import pandas as pd
import json
import time
import random

# ------------------- Load Data -------------------
# Load quiz CSV
try:
    quiz_data = pd.read_csv("algebra_ques_cleaned.csv")
except:
    try:
        quiz_data = pd.read_csv("algebra_ques.csv")
    except:
        st.error("Quiz data file missing!")
        st.stop()

# Load theory JSON (list of dicts)
try:
    with open("algebra_theory.json", "r", encoding="utf-8") as f:
        theory_data = json.load(f)
except:
    theory_data = []
    st.warning("No theory data found!")

# Load optional quiz JSON
try:
    with open("algebra_ques.json", "r", encoding="utf-8") as f:
        json_quiz = json.load(f)
except:
    json_quiz = {}

def go_to(page_name):
    """Helper function to change page + sync sidebar"""
    st.session_state.page = page_name
    st.rerun()
    
import pickle

# ------------------- Load ML Models -------------------
try:
    rf_model = pickle.load(open("student_performance_model.pkl", "rb"))
   
except:
    rf_model = None
    
try:
    kmeans_model = pickle.load(open("question_cluster_model.pkl", "rb"))
    vectorizer = pickle.load(open("question_vectorizer.pkl", "rb"))
    
except:
    kmeans_model, vectorizer = None, None
    
try:
    q_table = pickle.load(open("q_table.pkl", "rb"))
    
except:
    q_table = None
    

# ------------------- Streamlit Config -------------------
st.set_page_config(page_title="AI Learning Assistant", layout="wide")
st.title(" Welcome to Learning with AI")

# ------------------- Sidebar Navigation -------------------
pages = ["Home", "Theory", "Quiz", "Feedback & Suggestions"]

# Maintain current page in session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar radio (with correct selection)
page = st.sidebar.radio("Go to:", pages, index=pages.index(st.session_state.page))

# Update session state when sidebar changes
if page != st.session_state.page:
    st.session_state.page = page
    st.rerun()



# ------------------- Home Page -------------------
if page == "Home":
    st.subheader("Let's Begin Your AI-Powered Learning Journey")

    choice = st.radio("Do you want me to prepare a study timetable for you?", ["Yes", "No"])

    if choice == "Yes":
       st.markdown("### Let's Create Your Personalized Study Timetable")

    # Ask for inputs
       num_days = st.number_input("How many days do you have to study?", min_value=1, max_value=30, value=5)
       minutes_per_day = st.number_input("How many minutes can you study each day?", min_value=10, max_value=180, value=60)

       if st.button("Generate Timetable"):
          if not theory_data:
             st.warning("No theory data found to create timetable.")
          else:
             num_topics = len(theory_data)
             topics_per_day = max(1, num_topics // num_days)
             timetable = []

             topic_index = 0
             for day in range(1, num_days + 1):
                day_topics = [theory_data[i]["Topic"] for i in range(topic_index, min(topic_index + topics_per_day, num_topics))]
                topic_index += topics_per_day
                timetable.append({
                    "Day": day,
                    "Topics": day_topics,
                    "Study_Time": f"{minutes_per_day} minutes"
                })

             st.session_state["timetable"] = timetable
             st.session_state["timetable_mode"] = True

             st.success(" Timetable Created Successfully!")

             # Display timetable neatly
             st.markdown("### Your Study Timetable")
             for day_plan in timetable:
                with st.expander(f"Day {day_plan['Day']}"):
                    st.write(f"**Topics:** {', '.join(day_plan['Topics'])}")
                    st.write(f"**Study Time:** {day_plan['Study_Time']}")

    else:
        st.info("You can directly explore Theory, Quiz, and Feedback.")
        st.session_state["timetable_mode"] = False
    st.markdown("---")
    if st.button("-> Go to Theory Section", key="btn_home_to_theory"):
        go_to("Theory")


   

# ------------------- Theory Page -------------------
elif page == "Theory":
    st.subheader("Theory Section")

    if not theory_data:
        st.warning("No theory data found in JSON file.")
    else:
        if "topic_index" not in st.session_state:
            st.session_state.topic_index = 0

        topic_info = theory_data[st.session_state.topic_index]

        st.markdown(f"###  {topic_info['Topic']}")
        st.write(f"**Chapter:** {topic_info['Chapter']}")
        st.write(f"**Theory:** {topic_info['Theory']}")
        st.write(f"**Example Question:** {topic_info['Example_Question']}")
        st.info(f"**Step-by-Step Solution:** {topic_info['Step_by_Step']}")
        st.warning(f" Tip: {topic_info['Tips']}")
        st.markdown(f" **Book Reference:** {topic_info['Book_Reference']}")

        # Embed YouTube link
        st.markdown(f"[ Watch Video]({topic_info['YouTube_Link']})")

        # Navigation buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅ Previous") and st.session_state.topic_index > 0:
                st.session_state.topic_index -= 1
                st.rerun()
        with col2:
            st.write(f"Topic {st.session_state.topic_index + 1}/{len(theory_data)}")
        with col3:
            if st.button("-> Next") and st.session_state.topic_index < len(theory_data) - 1:
                st.session_state.topic_index += 1
                st.rerun()
    #  Go to Quiz Page
    st.markdown("---")
    if st.button(" Go to Quiz Section", key="btn_theory_to_quiz"):
       go_to("Quiz")


# ------------------- Quiz Page -------------------
elif page == "Quiz":
    st.subheader("Quiz Section")

    # Randomly select 10 questions if not already done
    if "quiz_questions" not in st.session_state or st.session_state.get("quiz_done", False):
        if len(quiz_data) < 10:
            selected_qs = quiz_data.sample(len(quiz_data), random_state=None)
        else:
            selected_qs = quiz_data.sample(10, random_state=None)
        st.session_state.quiz_questions = selected_qs.reset_index(drop=True)
        st.session_state.quiz_index = 0
        st.session_state.quiz_answers = {}
        st.session_state.start_time = time.time()
        st.session_state.quiz_done = False

    quiz_df = st.session_state.quiz_questions
    total_qs = len(quiz_df)
    current_index = st.session_state.quiz_index
    current_q = quiz_df.iloc[current_index]

    # Timer
    elapsed = int(time.time() - st.session_state.start_time)
    st.caption(f" Time Elapsed: {elapsed} seconds")

    # Display question
    st.markdown(f"### Q{current_index + 1}. {current_q['Question']}")
    
    # Parse options
    try:
        options = eval(current_q["Options"]) if isinstance(current_q["Options"], str) else current_q["Options"]
    except Exception:
        options = ["Option 1", "Option 2", "Option 3", "Option 4"]

    # Show options with no preselection
    previous_answer = st.session_state.quiz_answers.get(current_index, None)
    user_answer = st.radio(
        "Select your answer:",
        options,
        index=None if previous_answer is None else options.index(previous_answer),
        key=f"q{current_index}_radio"
    )

    # Save answer if selected
    if user_answer:
        st.session_state.quiz_answers[current_index] = user_answer

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅ Previous") and current_index > 0:
            st.session_state.quiz_index -= 1
            st.rerun()

    with col2:
        if current_index < total_qs - 1:
            if st.button("-> Next"):
                st.session_state.quiz_index += 1
                st.rerun()
        else:
            if st.button(" Submit Quiz"):
                # Evaluate
                correct = 0
                total_marks = 0
                scored_marks = 0
                wrong_questions = []

                for i in range(total_qs):
                    actual = quiz_df.iloc[i]["Correct_Answer"]
                    marks = quiz_df.iloc[i]["Marks"]
                    user_ans = st.session_state.quiz_answers.get(i, None)
                    total_marks += marks
                    if user_ans == actual:
                        correct += 1
                        scored_marks += marks
                    else:
                        wrong_questions.append({
                            "Question": quiz_df.iloc[i]["Question"],
                            "Your_Answer": user_ans,
                            "Correct_Answer": actual
                        })

                score_percent = (scored_marks / total_marks) * 100
                st.session_state.quiz_score = round(score_percent, 2)
                st.session_state.quiz_done = True
                st.session_state.wrong_questions = wrong_questions
                st.session_state.category_results = quiz_df.assign(
                    Given_Answer=[st.session_state.quiz_answers.get(i, None) for i in range(total_qs)]
                )
                
                # Show results
                st.success(f" Quiz Submitted! Your Score: {score_percent:.2f}% ({correct}/{total_qs} correct)")
                if wrong_questions:
                    st.markdown("###  Questions You Got Wrong:")
                    for wq in wrong_questions:
                        st.write(f"**Q:** {wq['Question']}")
                        st.write(f"Your Answer: {wq['Your_Answer']}")
                        st.write(f"Correct Answer:  {wq['Correct_Answer']}")
                        st.divider()

                st.info("Go to the ' Feedback & Suggestions' page for your AI-based performance analysis.")

# ------------------- Feedback & Suggestions -------------------
# ------------------- Feedback & Suggestions -------------------
elif page == "Feedback & Suggestions":
    st.subheader("AI Feedback and Personalized Learning Suggestions")

    # Check if quiz is done
    if not st.session_state.get("quiz_done", False):
        st.warning("Please complete the quiz to get feedback!")
    else:
        quiz_df = st.session_state.category_results
        score_percent = st.session_state.quiz_score

        # ------------------- Model 1: Random Forest -------------------
        # Calculate avg_time and avg_marks from quiz
        total_time = time.time() - st.session_state.start_time  # total time in seconds
        avg_time_per_q = total_time / len(quiz_df)

        total_marks = quiz_df['Marks'].sum()
        scored_marks = sum([
            quiz_df.iloc[i]["Marks"] 
            for i in range(len(quiz_df)) 
            if st.session_state.quiz_answers.get(i) == quiz_df.iloc[i]["Correct_Answer"]
        ])
        avg_marks_per_q = scored_marks / len(quiz_df)

        # Predict performance using RF model
        if rf_model:
            import numpy as np
            rf_pred = rf_model.predict(np.array([[avg_time_per_q, avg_marks_per_q]]))[0]
            st.markdown("### Type of questions you are lagging")
            st.info(rf_pred)
        else:
            st.warning("Random Forest model not loaded.")

        # ------------------- Display Quiz Score -------------------
        st.markdown("### Your Quiz Performance")
        st.write(f"**Score:** {score_percent:.2f}%")
        correct_count = sum([
            1 for i in range(len(quiz_df)) 
            if st.session_state.quiz_answers.get(i) == quiz_df.iloc[i]["Correct_Answer"]
        ])
        st.write(f"**Correct Answers:** {correct_count}/{len(quiz_df)}")

        # ------------------- Model 2: KMeans Recommendations -------------------
        if kmeans_model and vectorizer:
            if st.session_state.wrong_questions:
                # Recommend similar questions for first wrong question
                wrong_q_text = st.session_state.wrong_questions[0]["Question"]
                X = vectorizer.transform([wrong_q_text])
                cluster = kmeans_model.predict(X)[0]

                all_X = vectorizer.transform(quiz_df["Question"])
                labels = kmeans_model.predict(all_X)
                similar_qs = quiz_df[labels == cluster]["Question"].sample(
                    min(3, len(quiz_df[labels == cluster]))
                ).tolist()

                st.markdown("### Recommended Practice Questions (Similar to Mistakes)")
                for q in similar_qs:
                    st.write(f"• {q}")
        else:
            st.warning("Clustering model not loaded.")

        # ------------------- Model 3: Q-Learning Path -------------------
        import pickle
        try:
            with open("q_table.pkl", "rb") as f:
                q_table = pickle.load(f)
        except:
            q_table = None

        if q_table and isinstance(q_table, dict) and len(q_table) > 0:
            if score_percent >= 80:
                level = "excellent"
            elif score_percent >= 50:
                level = "good"
            else:
                level = "needs improvement"

            actions = q_table.get(level, {})
            if actions:
                next_step = max(actions, key=actions.get)
                st.markdown("### Personalized Learning Path (Q-Learning)")
                st.success(f"Next Step → {next_step}")
            else:
                st.warning("No adaptive path found for your level.")
        else:
            st.warning("Adaptive learning model not loaded or empty.")


    # ------------------- Restart Button -------------------
    st.markdown("---")
    if st.button("Restart Learning Journey", key="btn_feedback_to_home"):
        st.session_state.page = "Home"
        st.rerun()

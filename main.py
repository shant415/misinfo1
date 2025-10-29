import os
import streamlit as st
import json
import re
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# --- 1. Constants and API Configuration ---

# OpenAI Model for fact-checking/evaluation
MODEL_NAME = "gpt-4o-mini"
# New system prompt for extracting claims from long text
SYSTEM_PROMPT_EXTRACTION = (
    "You are an expert news analyst. Your task is to identify the 3 most significant, verifiable, factual claims from the user-provided text. "
    "Return the result as a single JSON object with one key, 'claims', which is an array of strings. "
    "Each string must be a concise, standalone claim ready for external fact-checking. DO NOT include any commentary, explanations, or text outside the JSON object. Example: {'claims': ['Claim 1.', 'Claim 2.', 'Claim 3.']}"
)

SYSTEM_PROMPT_WITH_FACTS = (
    "You are an AI-powered misinformation checker. Your task is to evaluate the truthfulness of a user's claim. "
    "You have been provided with existing fact-checks from Google's Fact Check Explorer. "
    "Based on the provided fact-check data, evaluate the user's claim on a scale of 1-10. "
    "Provide a concise, neutral explanation (3 sentences max). Make sure to include links from the fact-check data. "
    "ALWAYS begin your answer with a Rating: (score)/10."
)

SYSTEM_PROMPT_FALLBACK = (
    "You are an AI-powered misinformation checker. Your task is to evaluate the truthfulness of a user's claim "
    "based on your vast general knowledge, as no dedicated fact-checks were found. "
    "Your response MUST start with a truth rating on a scale of 1-10. "
    "Provide a concise, neutral explanation (3 sentences max). Do NOT mention using Google Search. "
    "ALWAYS begin your answer with a Rating: (score)/10."
)


# Determine where to load the secrets from: st.secrets (Streamlit) or os.environ (.env)
try:
    # 1. Try to load from Streamlit's secrets
    OPENAI_KEY = st.secrets["openai_key"]
    GOOGLE_KEY = st.secrets["google_key"] # Now needs to be loaded again

except Exception:
    # 2. Fall back to environment variables
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    GOOGLE_KEY = os.getenv("GOOGLE_API_KEY") # Use the standard environment variable name

# Input validation check - check for both keys now
if not OPENAI_KEY or not GOOGLE_KEY:
    st.error("🚨 Configuration Error: Missing one or more API keys.")
    st.info("Please ensure your `.streamlit/secrets.toml` file contains both `openai_key` and `google_key` defined.")
    st.stop()

# Initialize OpenAI Client globally
client = OpenAI(api_key=OPENAI_KEY)

# --- 3. Google Fact Check API Function ---

def fetch_claims(query, api_key):
    """Fetch Google Fact Check claims using the dedicated API."""
    try:
        # Note: The 'developerKey' parameter is used when building the service
        service = build("factchecktools", "v1alpha1", developerKey=api_key)
        request = service.claims().search(query=query)
        return request.execute()
    except HttpError as err:
        st.error(f"❌ Google Fact Check API Error: The Google API Key may be invalid or you may be exceeding your daily quota: {err}")
    except Exception as e:
        st.error(f"⚠️ Unexpected error when calling Google API: {e}")
    return None

# --- 4. NEW: Article Claim Extraction Function ---

def extract_claims_from_text(article_text):
    """Uses OpenAI to extract 3 key claims from a long article."""
    st.info("🧠 Step 1: Article detected. Extracting 3 key claims using OpenAI...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_EXTRACTION},
                {"role": "user", "content": article_text}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        # The model is instructed to return JSON, so we attempt to parse it
        raw_json = response.choices[0].message.content
        claims_data = json.loads(raw_json)
        
        return claims_data.get('claims', [])
    except Exception as e:
        st.error(f"❌ Failed to extract claims from article. AI output was likely not perfect JSON: {e}")
        return []


# --- 5. OpenAI Core Analysis Function ---

def openai_response(user_input, google_fact_data):
    """Generates an evaluated response using OpenAI, grounding it with Google Fact Check data if available."""
    
    messages = []

    if google_fact_data and google_fact_data.get('claims'):
        # If facts are found, use the grounding prompt
        messages.append({"role": "system", "content": SYSTEM_PROMPT_WITH_FACTS})
        # Add the raw fact-check JSON to the user prompt for the model to analyze
        fact_data_str = json.dumps(google_fact_data, indent=2)
        
        user_prompt = (
            f"EVALUATE THE CLAIM: '{user_input}'\n\n"
            f"Here are the existing fact-checks for evaluation:\n{fact_data_str}"
        )
        messages.append({"role": "user", "content": user_prompt})
    else:
        # If no facts are found, use the general knowledge prompt
        messages.append({"role": "system", "content": SYSTEM_PROMPT_FALLBACK})
        messages.append({"role": "user", "content": f"EVALUATE THE CLAIM: '{user_input}'"})


    try:
        # Make the API Call
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=messages,
            temperature=0.0,
            max_tokens=350,
        )
        
        # Extract Text
        text = response.choices[0].message.content
        
        # The sources come from the Google Fact Check data
        sources = [] 
        if google_fact_data and google_fact_data.get('claims'):
            for claim in google_fact_data['claims']:
                for review in claim.get('claimReview', []):
                    sources.append({"uri": review.get('url'), "title": review.get('publisher', {}).get('name', 'Fact Check Source')})
        
        # Clean up sources list to remove duplicates or entries where URL is missing
        unique_sources = []
        seen_uris = set()
        for source in sources:
            if source['uri'] and source['uri'] not in seen_uris:
                unique_sources.append(source)
                seen_uris.add(source['uri'])

        return text, unique_sources

    except Exception as e:
        st.error(f"⚠️ OpenAI API Error: {e}")
        return None, None 

# --- 6. Helper Functions for UI Processing ---

def extract_text_and_rating(response_text):
    """Extracts the numerical rating and the cleaned text."""
    if not response_text:
        return None, None
    
    # Extract Rating (e.g., Rating: 8/10)
    rating_match = re.search(r'Rating:\s*(\d{1,2})/10', response_text, re.IGNORECASE)
    rating = int(rating_match.group(1)) if rating_match else None
    
    # Clean Text (remove the rating line)
    cleaned_text = re.sub(r"Rating:\s*(\d{1,2}/10)\s*", "", response_text, count=1, flags=re.IGNORECASE).strip()
    
    # FIX: Use lstrip() to remove leading periods or extra spaces from the analysis
    cleaned_text = cleaned_text.lstrip(' .')
    
    return rating, cleaned_text

def interpret_rating(rating):
    """Provides a human-readable interpretation of the 1-10 numerical rating."""
    if rating is None:
        return "❓ Unrated"
    elif rating <= 2:
        return "❌ False"
    elif rating <= 4:
        return "⚠️ Most Likely False"
    elif rating <= 6:
        return "❓ Mixed: Both True and False"
    elif rating <= 8:
        return "✅ Likely True"
    else:
        return "✅ True"
        
def display_single_claim_result(claim, rating, interpretation, result_text, sources):
    """Displays the result for a single claim in a clean format."""
    st.subheader(f"📝 Claim: {claim}")
    
    # Rating Visual
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f"## **{rating}/10**")
    with col2:
        st.markdown(f"## {interpretation}")
        st.progress(rating / 10)
    
    st.markdown("---")
    
    # Explanation
    st.markdown(f"**Analysis:**")
    st.write(result_text)

    # Sources (Citations from Google Fact Check API)
    st.markdown("#### 🔗 Sources (Cited by Fact Check Organizations)")
    
    if sources:
        for source in sources:
            st.markdown(f"- [{source.get('title', 'Source Link')}]({source['uri']})")
    else:
        st.info("No dedicated fact-checks were found in the Google Fact Check Explorer.")
    
    st.divider()

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# --- 7. Streamlit UI with navigation ---

st.set_page_config(page_title="Fact Checker", layout="wide", initial_sidebar_state="expanded")
st.sidebar.title("🕵️‍♂️ AI Fact Checker 🕵️‍♂️")
page = st.sidebar.radio("Go to", ["Fact Checker", "Past Claims"])

# --- Fact Checker Page ---
if page == "Fact Checker":
    st.title("🕵️‍♂️ AI-Powered Misinformation Checker 🕵️‍♂️")
    st.markdown("Paste a claim or a full article/text below. ")
    
    # START OF EDIT: Use columns to control the width of the text area
    col_search, col_spacer = st.columns([3, 2])
    
    with col_search:
        # Single text area handles both short claim and long article
        # UPDATED HEIGHT FROM 200 TO 100
        user_input = st.text_area("Enter text to fact-check:", height=100)
        
        # Determine mode based on length
        is_article_mode = len(user_input) > 200
        
        if is_article_mode:
            st.warning("📄 Article Mode: The AI will extract and check the 3 most significant claims.")
            
        if st.button("Check Fact", type="primary"):
            if not user_input.strip():
                st.warning("⚠️ You must enter text to check!")
                st.stop()
            
            # --- Fact-Checking Pipeline ---
            
            claims_to_check = []
            
            if is_article_mode:
                # Step A: Extract claims from the long text
                claims_to_check = extract_claims_from_text(user_input)
                
                if not claims_to_check:
                    st.error("❌ Failed to extract any claims from the article. Please try a different article or shorten it.")
                    st.stop()
                
                st.success(f"✅ Extracted {len(claims_to_check)} claims. Starting individual fact-checks.")
            else:
                # Step B: Single Claim Mode
                claims_to_check = [user_input]
                st.info("💬 Claim Mode: Fact-checking your single claim.")
                
            # --- Run Pipeline for Each Claim ---
            
            all_results = []
            for i, claim in enumerate(claims_to_check):
                st.subheader(f"Analyzing Claim {i+1} of {len(claims_to_check)}: **{claim[:50]}...**")
                
                with st.spinner(f"🔎 Step 1: Searching Google's Fact Check Explorer for claim {i+1}..."):
                    google_fact_data = fetch_claims(claim, GOOGLE_KEY)
                
                with st.spinner(f"🧠 Step 2: Analyzing data for claim {i+1}..."):
                    response_text, sources = openai_response(claim, google_fact_data)
                
                if response_text is None:
                    st.error(f"❌ Failed to get analysis for claim {i+1}.")
                    continue
                
                rating, result_text = extract_text_and_rating(response_text)
                
                if result_text and rating is not None:
                    interpretation = interpret_rating(rating)
                    
                    # Display the result for the individual claim
                    display_single_claim_result(claim, rating, interpretation, result_text, sources)

                    # Save the result to history
                    all_results.append({
                        "claim": claim,
                        "rating": rating,
                        "interpretation": interpretation,
                        "result": result_text, 
                        "sources": sources
                    })
            
            # Save the combined result to session history
            st.session_state.history.extend(all_results)
            st.success("Analysis Complete!")
            

# --- Past Claims Page ---
elif page == "Past Claims":
    st.title("📜 Past Fact-Checks")
    if st.session_state.history:
        st.markdown("### 🔎 Search Past Claims")
        search_term = st.text_input("Search claims:", key="search")
        
        # Filter and display history
        for entry in reversed(st.session_state.history):
            if search_term.lower() in entry["claim"].lower():
                with st.expander(f"📝 {entry['claim']} (Rating: {entry['rating']}/10)"):
                    st.markdown(f"**Interpretation:** {entry['interpretation']}")
                    st.markdown(f"**AI Analysis:** {entry['result']}")
                    
                    st.markdown("**Sources:**")
                    if entry.get('sources'):
                        for source in entry['sources']:
                            if source.get('uri'):
                                st.markdown(f"- [{source.get('title', source['uri'])}]({source['uri']})")
                    else:
                        st.info("No saved sources for this claim.")
        
        st.divider()
        st.download_button(
            label="📥 Download History",
            data=json.dumps(st.session_state.history, indent=2),
            file_name="fact_check_history.json",
            mime="application/json"
        )

        if st.button("🧹 Clear History"):
            st.session_state.history.clear()
            st.rerun() 
            st.success("History cleared!")
    else:
        st.info("No past fact-checks available.")

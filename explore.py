import re
import pandas as pd
from typing import Dict, List, Optional

# Define keywords for each intervention category with more comprehensive coverage
CATEGORIES: Dict[str, List[str]] = {
    "Physical Activity - Cardio": [
        "run", "running", "jog", "jogging", "cycling", "bike", "spin class",
        "hiit", "cardio", "treadmill", "exercise session", "gym session",
        "swimming", "aerobic", "dance", "zumba"
    ],
    
    "Physical Activity - Non-cardio": [
        "walk", "walking", "yoga", "stretch", "light exercise", "light yoga",
        "pilates", "tai chi", "meditation", "breathing exercise", "strength training",
        "weights", "resistance"
    ],
    
    "Social Media & Screen Time": [
        "scrolling phone", "social media", "facebook", "instagram", "tiktok",
        "twitter", "youtube", "media consumption", "screen time", "browsing",
        "reddit", "linkedin", "online", "internet"
    ],
    
    "Rest & Recovery": [
        "nap", "rest", "sleep", "break", "relaxation", "meditation",
        "mindfulness", "quiet time", "recharge"
    ],
    
    "Nutrition & Consumption": [
        "eating", "drinking", "coffee", "lunch", "breakfast", "dinner", "meal",
        "snack", "tea", "water", "hydration", "nutrition", "food"
    ],
    
    "Social Interaction": [
        "call", "social", "1-1", "meeting", "friends", "conversation", "hangout",
        "team meeting", "mentor", "office hours", "chat", "discussion", "talk",
        "collaboration", "group work", "networking", "socializing"
    ],
    
    "Mindful Activities": [
        "journal", "journaling", "writing", "gratitude", "gratitude journal",
        "reflection", "diary", "planning", "goal setting", "mindful writing"
    ],
    
    "Learning & Reading": [
        "read", "reading", "book", "devotional", "study", "learning",
        "educational", "research", "article", "paper", "textbook", "literature"
    ],
    
    "Work & Productivity": [
        "work", "coding", "programming", "writing", "analysis", "research",
        "project", "task", "deadline", "focus session", "deep work"
    ]
}

def categorize_intervention(name: str, description: str = "", confidence_threshold: float = 0.5) -> Dict[str, any]:
    """
    Attempts to classify an intervention/event based on the name and description.
    """
    text = f"{name} {description}".lower()
    
    best_match = {
        "category": "Other",
        "confidence": 0.0,
        "matched_keywords": []
    }
    
    for category, keywords in CATEGORIES.items():
        matched_keywords = [kw for kw in keywords if kw in text]
        if matched_keywords:
            confidence = len(matched_keywords) / len(keywords)
            confidence += sum(len(kw) for kw in matched_keywords) / (100 * len(keywords))
            
            if confidence > best_match["confidence"]:
                best_match = {
                    "category": category,
                    "confidence": min(confidence, 1.0),
                    "matched_keywords": matched_keywords
                }
    
    if best_match["confidence"] < confidence_threshold:
        best_match["category"] = "Other"
    
    return best_match

def analyze_intervention_data(df: pd.DataFrame, 
                            name_col: str,
                            description_col: Optional[str] = None) -> pd.DataFrame:
    """
    Analyzes a DataFrame of interventions and adds categorization information.
    """
    results = []
    for _, row in df.iterrows():
        description = row.get(description_col, "") if description_col else ""
        categorization = categorize_intervention(
            name=str(row[name_col]),
            description=str(description)
        )
        results.append(categorization)
    
    df['category'] = [r['category'] for r in results]
    df['categorization_confidence'] = [r['confidence'] for r in results]
    df['matched_keywords'] = [r['matched_keywords'] for r in results]
    
    return df 
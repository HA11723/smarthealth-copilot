# Main Streamlit application for SmartHealth Copilot
import streamlit as st
import os
import sys
from dotenv import load_dotenv
import json
from openai import OpenAI
from typing import List, Dict, Any, Optional
import io
from google.cloud import vision

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()


def load_medical_data():
    """Load medical cases and symptoms data"""
    try:
        with open('app/data/medical_cases.json', 'r') as f:
            medical_cases = json.load(f)
        with open('app/data/symptoms_db.json', 'r') as f:
            symptoms_db = json.load(f)
        return medical_cases, symptoms_db
    except FileNotFoundError:
        return [], []


def find_similar_cases(user_symptoms: List[str], medical_cases: List[Dict]) -> List[Dict]:
    """Find similar medical cases based on symptoms"""
    similar_cases = []
    user_symptoms_lower = [s.lower() for s in user_symptoms]

    for case in medical_cases:
        case_symptoms_lower = [s.lower() for s in case['symptoms']]

        # Check for symptom overlap
        matches = 0
        for user_symptom in user_symptoms_lower:
            for case_symptom in case_symptoms_lower:
                if user_symptom in case_symptom or case_symptom in user_symptom:
                    matches += 1

        # If there are matches, add to similar cases
        if matches > 0:
            similarity_score = matches / len(case_symptoms_lower)
            similar_cases.append({
                **case,
                'similarity_score': similarity_score,
                'matching_symptoms': matches
            })

    # Sort by similarity score
    similar_cases.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similar_cases[:3]  # Return top 3 similar cases


def analyze_symptom_urgency(user_symptoms: List[str], symptoms_db: List[Dict]) -> Dict[str, Any]:
    """Analyze symptom urgency using local database"""
    urgency_analysis = {
        'high_urgency_symptoms': [],
        'medium_urgency_symptoms': [],
        'low_urgency_symptoms': [],
        'overall_urgency': 'LOW'
    }

    user_symptoms_lower = [s.lower() for s in user_symptoms]

    for symptom_entry in symptoms_db:
        symptom_name = symptom_entry['symptom'].lower()

        # Check if user has this symptom
        for user_symptom in user_symptoms_lower:
            if symptom_name in user_symptom or user_symptom in symptom_name:
                # Check urgency levels
                for urgency_level, descriptions in symptom_entry['urgency_levels'].items():
                    for description in descriptions:
                        if any(word in user_symptom.lower() for word in description.lower().split()):
                            if urgency_level == 'high':
                                urgency_analysis['high_urgency_symptoms'].append(
                                    symptom_name)
                            elif urgency_level == 'medium':
                                urgency_analysis['medium_urgency_symptoms'].append(
                                    symptom_name)
                            else:
                                urgency_analysis['low_urgency_symptoms'].append(
                                    symptom_name)

    # Determine overall urgency
    if urgency_analysis['high_urgency_symptoms']:
        urgency_analysis['overall_urgency'] = 'HIGH'
    elif urgency_analysis['medium_urgency_symptoms']:
        urgency_analysis['overall_urgency'] = 'MEDIUM'

    return urgency_analysis


class MedicalLLM:
    """LLM integration for medical analysis using OpenAI"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    def analyze_symptoms(self, age: int, gender: str, symptoms: str, additional_symptoms: List[str],
                         similar_cases: Optional[List[Dict]] = None, urgency_analysis: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze symptoms using OpenAI with enhanced context"""

        # Set default values
        if similar_cases is None:
            similar_cases = []
        if urgency_analysis is None:
            urgency_analysis = {}

        # Detect input language
        input_language = self._detect_language(
            symptoms + " " + " ".join(additional_symptoms))

        # Create the prompt with additional context
        prompt = self._create_enhanced_medical_prompt(age, gender, symptoms, additional_symptoms,
                                                      similar_cases, urgency_analysis, input_language)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(
                        input_language)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            # Parse the response
            content = response.choices[0].message.content
            if content:
                analysis = self._parse_llm_response(content)
            else:
                analysis = self._fallback_parsing(
                    "No response from LLM", input_language)

            # Enhance with local data
            if similar_cases:
                analysis['local_similar_cases'] = similar_cases
            if urgency_analysis:
                analysis['local_urgency_analysis'] = urgency_analysis

            return analysis

        except Exception as e:
            return {
                "error": f"LLM analysis failed: {str(e)}",
                "similar_cases": [],
                "possible_diagnoses": [],
                "urgency_level": "UNKNOWN",
                "recommendations": []
            }

    def _detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        # Simple language detection based on character sets
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([char for char in text if char.isalpha()])

        if total_chars > 0 and arabic_chars / total_chars > 0.3:
            return "arabic"
        elif any('\u4e00' <= char <= '\u9fff' for char in text):  # Chinese characters
            return "chinese"
        elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):  # Japanese
            return "japanese"
        elif any('\uac00' <= char <= '\ud7af' for char in text):  # Korean
            return "korean"
        elif any('\u0900' <= char <= '\u097f' for char in text):  # Hindi/Devanagari
            return "hindi"
        else:
            return "english"

    def _get_system_prompt(self, language: str = "english") -> str:
        """Get the system prompt for medical analysis in the specified language"""
        if language == "arabic":
            return """أنت مساعد طبي ذكي. دورك هو:
1. تحليل الأعراض وتقديم التشخيصات المحتملة
2. تقييم مستويات الطوارئ
3. تقديم التوصيات الطبية
4. التأكيد دائماً على استشارة المتخصصين في الرعاية الصحية

مهم: هذا للأغراض التعليمية فقط. استشر دائماً متخصصين في الرعاية الصحية للحصول على النصائح الطبية الفعلية.

أجب بتنسيق JSON بالهيكل التالي:
{
    "similar_cases": ["حالة1", "حالة2"],
    "possible_diagnoses": [
        {"diagnosis": "الحالة", "probability": "عالية/متوسطة/منخفضة", "confidence": "السبب"}
    ],
    "urgency_level": "عالية/متوسطة/منخفضة",
    "recommendations": ["توصية1", "توصية2", "توصية3"]
}"""
        else:
            return """You are a medical AI assistant. Your role is to:
1. Analyze symptoms and provide possible diagnoses
2. Assess urgency levels
3. Provide medical recommendations
4. Always emphasize consulting healthcare professionals

IMPORTANT: This is for educational purposes only. Always recommend consulting healthcare professionals for actual medical advice.

Respond in JSON format with the following structure:
{
    "similar_cases": ["case1", "case2"],
    "possible_diagnoses": [
        {"diagnosis": "condition", "probability": "high/medium/low", "confidence": "reason"}
    ],
    "urgency_level": "HIGH/MEDIUM/LOW",
    "recommendations": ["rec1", "rec2", "rec3"]
}"""

    def _create_enhanced_medical_prompt(self, age: int, gender: str, symptoms: str,
                                        additional_symptoms: List[str], similar_cases: Optional[List[Dict]] = None,
                                        urgency_analysis: Optional[Dict] = None, language: str = "english") -> str:
        """Create enhanced medical analysis prompt with local data context"""
        all_symptoms = [symptoms] + additional_symptoms
        symptoms_text = ", ".join(all_symptoms)

        if language == "arabic":
            prompt = f"""حلل الحالة الطبية التالية:

معلومات المريض:
- العمر: {age}
- الجنس: {gender}
- الأعراض: {symptoms_text}

يرجى تقديم:
1. حالات طبية مشابهة قد تكون ذات صلة
2. التشخيصات المحتملة مع مستويات الاحتمالية
3. تقييم الطوارئ (عالية/متوسطة/منخفضة)
4. توصيات محددة للخطوات التالية

ركز على أعراض القلب والأوعية الدموية والجهاز التنفسي والعصبية لأنها الأكثر أهمية."""
        else:
            prompt = f"""Analyze the following medical case:

Patient Information:
- Age: {age}
- Gender: {gender}
- Symptoms: {symptoms_text}

Please provide:
1. Similar medical cases that might be relevant
2. Possible diagnoses with probability levels
3. Urgency assessment (HIGH/MEDIUM/LOW)
4. Specific recommendations for next steps

Focus on cardiovascular, respiratory, and neurological symptoms as these are most critical."""

        # Add local data context if available
        if similar_cases and len(similar_cases) > 0:
            if language == "arabic":
                prompt += f"\n\nالحالات المشابهة من قاعدة البيانات الطبية:"
                for i, case in enumerate(similar_cases[:2], 1):
                    prompt += f"\nالحالة {i}: {case['age']} سنة {case['gender']} مع {', '.join(case['symptoms'])} → {case['diagnosis']} (الطوارئ: {case['urgency']})"
            else:
                prompt += f"\n\nSimilar cases from medical database:"
                for i, case in enumerate(similar_cases[:2], 1):
                    prompt += f"\nCase {i}: {case['age']}-year-old {case['gender']} with {', '.join(case['symptoms'])} → {case['diagnosis']} (urgency: {case['urgency']})"

        if urgency_analysis and 'overall_urgency' in urgency_analysis:
            if language == "arabic":
                prompt += f"\n\nتحليل الطوارئ المحلي: مستوى الطوارئ {urgency_analysis['overall_urgency']}"
            else:
                prompt += f"\n\nLocal urgency analysis: {urgency_analysis['overall_urgency']} urgency level"

        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data"""
        try:
            # Try to parse as JSON
            if response.strip().startswith('{'):
                return json.loads(response)
            else:
                # Fallback parsing
                return self._fallback_parsing(response)
        except json.JSONDecodeError:
            return self._fallback_parsing(response)

    def _fallback_parsing(self, response: str, language: str = "english") -> Dict[str, Any]:
        """Fallback parsing for non-JSON responses"""
        if language == "arabic":
            return {
                "similar_cases": ["تم إكمال التحليل"],
                "possible_diagnoses": [
                    {"diagnosis": "يحتاج استشارة طبية",
                        "probability": "عالية", "confidence": "تحليل الذكاء الاصطناعي"}
                ],
                "urgency_level": "متوسطة",
                "recommendations": [
                    "استشر متخصص في الرعاية الصحية",
                    "راقب الأعراض عن كثب",
                    "اطلب الرعاية الفورية إذا ساءت الأعراض"
                ]
            }
        else:
            return {
                "similar_cases": ["Analysis completed"],
                "possible_diagnoses": [
                    {"diagnosis": "Medical consultation needed",
                        "probability": "high", "confidence": "AI analysis"}
                ],
                "urgency_level": "MEDIUM",
                "recommendations": [
                    "Consult with a healthcare professional",
                    "Monitor symptoms closely",
                    "Seek immediate care if symptoms worsen"
                ]
            }


def analyze_image_with_vision_api(image_bytes: bytes):
    """Send image bytes to Google Cloud Vision API and return labels/results."""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    results = []
    for label in labels:
        results.append({
            'description': label.description,
            'score': label.score
        })
    return results


def load_doctors_data():
    """Load doctors database"""
    try:
        with open('app/data/doctors.json', 'r') as f:
            doctors = json.load(f)
        return doctors
    except FileNotFoundError:
        return []


def match_symptoms_to_doctor(symptoms: List[str], doctors: List[Dict]) -> Optional[Dict]:
    """Match user symptoms to the most appropriate doctor"""
    if not doctors:
        return None

    # Create a mapping of symptoms to specialties
    symptom_specialty_mapping = {
        'headache': 'Headache Specialist',
        'migraine': 'Headache Specialist',
        'head pain': 'Headache Specialist',
        'skin': 'Dermatologist',
        'rash': 'Dermatologist',
        'acne': 'Dermatologist',
        'dermatology': 'Dermatologist',
        'bone': 'Orthopedic Surgeon',
        'joint': 'Orthopedic Surgeon',
        'fracture': 'Orthopedic Surgeon',
        'sprain': 'Orthopedic Surgeon',
        'orthopedic': 'Orthopedic Surgeon',
        'heart': 'Cardiologist',
        'chest pain': 'Cardiologist',
        'cardiac': 'Cardiologist',
        'child': 'Pediatrician',
        'pediatric': 'Pediatrician',
        'baby': 'Pediatrician',
        'infant': 'Pediatrician'
    }

    # Check user symptoms against the mapping
    user_symptoms_lower = [s.lower() for s in symptoms]
    matched_specialty = None

    for symptom in user_symptoms_lower:
        for key, specialty in symptom_specialty_mapping.items():
            if key in symptom or symptom in key:
                matched_specialty = specialty
                break
        if matched_specialty:
            break

    # Find the doctor with the matched specialty
    if matched_specialty:
        for doctor in doctors:
            if doctor['specialty'] == matched_specialty:
                return doctor

    return None


def main():
    # Page configuration
    st.set_page_config(
        page_title="🧠 SmartHealth Copilot",
        page_icon="🧠",
        layout="wide"
    )

    # Main title and description
    st.title("🧠 SmartHealth Copilot")
    st.markdown("### Your AI-Powered Medical Assistant")
    st.markdown("---")

    st.header("\U0001F4F7 Injury/Condition Image Analysis (Beta)")
    uploaded_file = st.file_uploader(
        "Upload an image of your injury or condition (e.g., arm, leg, skin)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        try:
            st.image(image_bytes, caption="Uploaded Image",
                     use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {e}")
            st.info("Image uploaded successfully but could not be displayed.")
        with st.spinner("Analyzing image with AI..."):
            try:
                results = analyze_image_with_vision_api(image_bytes)
                if results:
                    st.success("AI Analysis Results:")
                    for res in results:
                        st.write(
                            f"- **{res['description']}** (confidence: {res['score']:.2f})")
                else:
                    st.warning("No significant findings detected.")
            except Exception as e:
                st.error(f"Image analysis failed: {e}")

    # Load medical data
    medical_cases, symptoms_db = load_medical_data()
    doctors = load_doctors_data()

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("API Keys are loaded from .env file")

        # Display API status
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")

        if openai_key and openai_key != "your_openai_api_key_here":
            st.success("✅ OpenAI API Key: Configured")
        else:
            st.error("❌ OpenAI API Key: Not configured")

        if pinecone_key and pinecone_key != "your_pinecone_api_key_here":
            st.success("✅ Pinecone API Key: Configured")
        else:
            st.error("❌ Pinecone API Key: Not configured")

        # Show data status
        if medical_cases:
            st.success(f"✅ Medical Cases: {len(medical_cases)} loaded")
        else:
            st.warning("⚠️ Medical Cases: Not loaded")

        if symptoms_db:
            st.success(
                f"✅ Symptoms Database: {len(symptoms_db)} symptoms loaded")
        else:
            st.warning("⚠️ Symptoms Database: Not loaded")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📝 Enter Your Symptoms")

        # Patient information
        st.subheader("Patient Information")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        # Symptoms input
        st.subheader("Symptoms & Medical History")
        symptoms = st.text_area(
            "Describe your symptoms or medical history:",
            placeholder="Example: I'm a 45-year-old male experiencing chest tightness and shortness of breath for the past 2 hours...",
            height=150
        )

        # Additional symptoms
        st.subheader("Additional Symptoms (Optional)")
        additional_symptoms = st.multiselect(
            "Select any additional symptoms:",
            ["Fever", "Headache", "Nausea", "Fatigue", "Dizziness", "Chest Pain",
             "Shortness of Breath", "Cough", "Sore Throat", "Joint Pain", "Rash"]
        )

        # Analysis button
        if st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True):
            if symptoms.strip():
                analyze_symptoms(age, gender, symptoms,
                                 additional_symptoms, medical_cases, symptoms_db, doctors)
            else:
                st.warning("Please enter your symptoms to analyze.")

    with col2:
        st.header("📊 Analysis Results")
        st.info("Results will appear here after analysis")

        # Placeholder for results
        st.markdown("""
        ### What to expect:
        - **Similar Cases**: Previous medical cases with similar symptoms
        - **Possible Diagnoses**: AI-generated diagnostic suggestions
        - **Urgency Level**: Assessment of symptom severity
        - **Recommendations**: Next steps and advice
        """)

        # Disclaimer
        st.markdown("---")
        st.caption("""
        ⚠️ **Disclaimer**: This is an AI-powered analysis for educational purposes only. 
        Always consult with a qualified healthcare professional for medical advice.
        """)


def analyze_symptoms(age, gender, symptoms, additional_symptoms, medical_cases, symptoms_db, doctors):
    """Enhanced symptom analysis using OpenAI and local data"""

    # Initialize LLM
    llm = MedicalLLM()

    # Prepare all symptoms
    all_symptoms = [symptoms] + additional_symptoms

    # Find similar cases from local database
    similar_cases = find_similar_cases(all_symptoms, medical_cases)

    # Analyze urgency using local database
    urgency_analysis = analyze_symptom_urgency(all_symptoms, symptoms_db)

    # Show loading spinner
    with st.spinner("🤖 AI is analyzing your symptoms..."):
        # Get enhanced analysis from OpenAI
        analysis = llm.analyze_symptoms(age, gender, symptoms, additional_symptoms,
                                        similar_cases, urgency_analysis)

    st.success("✅ Analysis Complete!")

    # Display results
    with st.container():
        st.subheader("🔍 Analysis Results")

        # Check for errors
        if "error" in analysis:
            st.error(f"❌ Analysis Error: {analysis['error']}")
            return

        # Local similar cases
        if analysis.get("local_similar_cases"):
            st.markdown("**📋 Similar Cases from Database:**")
            for case in analysis["local_similar_cases"]:
                st.info(
                    f"• **{case['age']}-year-old {case['gender']}**: {', '.join(case['symptoms'])} → {case['diagnosis']} (Urgency: {case['urgency']})")

        # AI similar cases
        if analysis.get("similar_cases"):
            st.markdown("**🤖 AI-Generated Similar Cases:**")
            for case in analysis["similar_cases"]:
                st.info(f"• {case}")

        # Possible diagnoses
        if analysis.get("possible_diagnoses"):
            st.markdown("**🏥 Possible Diagnoses:**")
            for diagnosis in analysis["possible_diagnoses"]:
                prob_level = diagnosis.get("probability", "unknown")
                confidence = diagnosis.get("confidence", "")
                st.warning(
                    f"• **{diagnosis.get('diagnosis', 'Unknown')}** ({prob_level})")
                if confidence:
                    st.caption(f"  *{confidence}*")

        # Urgency assessment
        urgency_level = analysis.get("urgency_level", "UNKNOWN")
        local_urgency = analysis.get("local_urgency_analysis", {}).get(
            "overall_urgency", "UNKNOWN")

        st.markdown("**⚠️ Urgency Assessment:**")

        # Show both AI and local urgency
        if urgency_level != "UNKNOWN":
            st.info(f"🤖 AI Assessment: {urgency_level}")
        if local_urgency != "UNKNOWN":
            st.info(f"📊 Database Assessment: {local_urgency}")

        # Overall urgency (use the higher one)
        final_urgency = urgency_level if urgency_level != "UNKNOWN" else local_urgency

        if final_urgency == "HIGH":
            st.error(f"🚨 OVERALL URGENCY: {final_urgency}")
            st.markdown("**Immediate medical attention recommended.**")
        elif final_urgency == "MEDIUM":
            st.warning(f"⚠️ OVERALL URGENCY: {final_urgency}")
            st.markdown(
                "**Monitor symptoms and consult healthcare provider.**")
        else:
            st.info(f"ℹ️ OVERALL URGENCY: {final_urgency}")
            st.markdown("**Continue monitoring symptoms.**")

        # Recommendations
        if analysis.get("recommendations"):
            st.markdown("**💡 Recommendations:**")
            for i, rec in enumerate(analysis["recommendations"], 1):
                st.info(f"{i}. {rec}")

        # Match to appropriate doctor
        matched_doctor = match_symptoms_to_doctor(
            [symptoms] + additional_symptoms, doctors)

        if matched_doctor:
            st.success(
                f"🎯 **Recommended Specialist:** {matched_doctor['name']}")
            st.info(f"**Specialty:** {matched_doctor['specialty']}")
            st.info(f"**Phone:** {matched_doctor['phone']}")
            st.info(f"**Email:** {matched_doctor['email']}")
            st.info(f"**Availability:** {matched_doctor['availability']}")
        else:
            st.warning(
                "No specific specialist matched. Consider consulting a general practitioner.")

        # Disclaimer
        st.markdown("---")
        st.caption("""
        ⚠️ **Disclaimer**: This is an AI-powered analysis for educational purposes only. 
        Always consult with a qualified healthcare professional for medical advice.
        """)


if __name__ == "__main__":
    main()

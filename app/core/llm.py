import os
from openai import OpenAI
from typing import List, Dict, Any
import json


class MedicalLLM:
    """LLM integration for medical analysis using OpenAI"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")

    def analyze_symptoms(self, age: int, gender: str, symptoms: str, additional_symptoms: List[str]) -> Dict[str, Any]:
        """Analyze symptoms using OpenAI"""

        # Create the prompt
        prompt = self._create_medical_prompt(
            age, gender, symptoms, additional_symptoms)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
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
                analysis = self._fallback_parsing("No response from LLM")
            return analysis

        except Exception as e:
            return {
                "error": f"LLM analysis failed: {str(e)}",
                "similar_cases": [],
                "possible_diagnoses": [],
                "urgency_level": "UNKNOWN",
                "recommendations": []
            }

    def _get_system_prompt(self) -> str:
        """Get the system prompt for medical analysis"""
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

    def _create_medical_prompt(self, age: int, gender: str, symptoms: str, additional_symptoms: List[str]) -> str:
        """Create the medical analysis prompt"""
        all_symptoms = [symptoms] + additional_symptoms
        symptoms_text = ", ".join(all_symptoms)

        return f"""Analyze the following medical case:

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

    def _fallback_parsing(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for non-JSON responses"""
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

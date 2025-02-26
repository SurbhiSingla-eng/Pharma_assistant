import os
import re
import json
import tempfile
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Image processing
import pytesseract
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import requests

# Google APIs for storage
import gspread
from google.oauth2.service_account import Credentials

# System constants
SERVICE_ACCOUNT_FILE = 'service_account.json'
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]
PRESCRIPTION_SHEET = "Prescription_Log"

# Mock database of medications (in a real system, this would be a proper database)
MEDICATION_DATABASE = {
    "aspirin": {
        "dosages": ["81mg", "325mg", "500mg"],
        "forms": ["tablet", "capsule", "liquid"],
        "instructions": ["daily", "twice daily", "with food"],
        "ndc_codes": ["00904-2013-61", "00904-2015-61"],
        "price": 5.99
    },
    "ibuprofen": {
        "dosages": ["200mg", "400mg", "600mg", "800mg"],
        "forms": ["tablet", "capsule", "liquid"],
        "instructions": ["daily", "twice daily", "with food", "as needed for pain"],
        "ndc_codes": ["00904-5755-61", "00904-5756-61"],
        "price": 7.99
    },
    "amoxicillin": {
        "dosages": ["250mg", "500mg", "875mg"],
        "forms": ["tablet", "capsule", "liquid"],
        "instructions": ["daily", "twice daily", "three times daily", "with food"],
        "ndc_codes": ["00904-5790-61", "00904-5791-61"],
        "price": 12.99
    },
    "lisinopril": {
        "dosages": ["5mg", "10mg", "20mg", "40mg"],
        "forms": ["tablet"],
        "instructions": ["daily", "in the morning", "with or without food"],
        "ndc_codes": ["00904-5940-61", "00904-5941-61"],
        "price": 9.99
    },
    "metformin": {
        "dosages": ["500mg", "850mg", "1000mg"],
        "forms": ["tablet", "extended-release tablet"],
        "instructions": ["daily", "twice daily", "with meals"],
        "ndc_codes": ["00904-6070-61", "00904-6071-61"],
        "price": 8.99
    }
}

# Mock database of symptoms and conditions (in a real system, this would be much more comprehensive)
SYMPTOM_DATABASE = {
    "fever": ["common cold", "flu", "covid-19", "infection", "pneumonia"],
    "cough": ["common cold", "flu", "covid-19", "pneumonia", "bronchitis"],
    "headache": ["migraine", "tension headache", "sinusitis", "concussion", "dehydration"],
    "fatigue": ["anemia", "depression", "hypothyroidism", "sleep apnea", "chronic fatigue syndrome"],
    "rash": ["eczema", "psoriasis", "allergic reaction", "chickenpox", "measles"],
    "chest pain": ["angina", "heart attack", "pulmonary embolism", "pneumonia", "acid reflux"],
    "shortness of breath": ["asthma", "copd", "heart failure", "pneumonia", "anxiety"],
    "abdominal pain": ["appendicitis", "gallstones", "ibs", "gastritis", "food poisoning"]
}


class PrescriptionProcessor:
    """Handles prescription image processing, text extraction, and medication matching"""
    
    def __init__(self, tesseract_path=r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        """Initialize the prescription processor with path to Tesseract OCR"""
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Common medication patterns for regex matching
        self.medication_patterns = {
            r'(?i)aspirin\s*(\d+)\s*mg': ("aspirin", "{0}mg"),
            r'(?i)ibuprofen\s*(\d+)\s*mg': ("ibuprofen", "{0}mg"),
            r'(?i)amoxicillin\s*(\d+)\s*mg': ("amoxicillin", "{0}mg"),
            r'(?i)lisinopril\s*(\d+)\s*mg': ("lisinopril", "{0}mg"),
            r'(?i)metformin\s*(\d+)\s*mg': ("metformin", "{0}mg"),
            # Add more patterns as needed
        }
        
        # Instruction patterns
        self.instruction_patterns = [
            r'(?i)(\d+)x\s*daily',
            r'(?i)(\d+)\s*times?\s*a\s*day',
            r'(?i)(once|twice|three\s*times)\s*daily',
            r'(?i)with\s*meals?',
            r'(?i)before\s*meals?',
            r'(?i)after\s*meals?',
            r'(?i)as\s*needed',
            r'(?i)every\s*(\d+)\s*hours?',
        ]
    
    def process_image_url(self, image_url: str) -> Tuple[str, str, Dict]:
        """Process a prescription image from a URL"""
        # Download image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Download the image
            response = requests.get(image_url)
            response.raise_for_status()
            
            with open(temp_path, "wb") as f:
                f.write(response.content)
            
            # Process the image
            extracted_text = self.extract_text_from_image(temp_path)
            medication_details = self.match_medication(extracted_text)
            
            return extracted_text, medication_details["name"], medication_details
        
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def process_image_file(self, image_path: str) -> Tuple[str, str, Dict]:
        """Process a prescription image from a local file"""
        extracted_text = self.extract_text_from_image(image_path)
        medication_details = self.match_medication(extracted_text)
        
        return extracted_text, medication_details["name"], medication_details
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using OCR with enhanced preprocessing"""
        try:
            # Read the image with OpenCV for preprocessing
            img = cv2.imread(image_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to handle different lighting conditions
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Noise removal with dilate/erode operations
            kernel = np.ones((1, 1), np.uint8)
            img_processed = cv2.dilate(thresh, kernel, iterations=1)
            img_processed = cv2.erode(img_processed, kernel, iterations=1)
            
            # Convert back to PIL Image for Tesseract
            pil_img = Image.fromarray(img_processed)
            
            # Extract text with custom config
            text = pytesseract.image_to_string(
                pil_img, 
                config='--oem 1 --psm 6 -l eng'
            ).strip()
            
            return text or "No text detected"
        
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            return f"OCR Error: {str(e)}"
    
    def match_medication(self, text: str) -> Dict:
        """Match medication names, dosages, and instructions in the extracted text"""
        text = text.lower()
        result = {
            "name": "unknown",
            "dosage": "unknown",
            "instructions": "unknown",
            "confidence": 0.0,
            "matched_text": "",
            "price": 0.0,
            "ndc_code": "",
            "form": "tablet"
        }
        
        # Try to match medication names and dosages
        for pattern, (med_name, dosage_template) in self.medication_patterns.items():
            for match in re.finditer(pattern, text):
                if match and match.groups():
                    dosage = match.group(1)
                    result["name"] = med_name
                    result["dosage"] = dosage_template.format(dosage)
                    result["matched_text"] = match.group(0)
                    result["confidence"] = 0.8
                    
                    # Get additional information from our database
                    if med_name in MEDICATION_DATABASE:
                        med_info = MEDICATION_DATABASE[med_name]
                        result["price"] = med_info["price"]
                        result["ndc_code"] = med_info["ndc_codes"][0] if med_info["ndc_codes"] else ""
                        result["form"] = med_info["forms"][0] if med_info["forms"] else "tablet"
        
        # If we haven't found a match with dosage, try simple keyword matching
        if result["name"] == "unknown":
            for med_name in MEDICATION_DATABASE.keys():
                if med_name in text:
                    result["name"] = med_name
                    result["matched_text"] = med_name
                    result["confidence"] = 0.6
                    
                    # Get additional information from our database
                    med_info = MEDICATION_DATABASE[med_name]
                    result["price"] = med_info["price"]
                    result["ndc_code"] = med_info["ndc_codes"][0] if med_info["ndc_codes"] else ""
                    result["form"] = med_info["forms"][0] if med_info["forms"] else "tablet"
                    result["dosage"] = med_info["dosages"][0] if med_info["dosages"] else "unknown"
        
        # Try to match instructions
        for pattern in self.instruction_patterns:
            for match in re.finditer(pattern, text):
                if match:
                    result["instructions"] = match.group(0)
                    break
        
        return result


class DiagnosticAnalyzer:
    """Handles diagnostic image analysis and symptom processing"""
    
    def __init__(self):
        # In a real system, this would load trained medical image analysis models
        pass
    
    def analyze_medical_image(self, image_path: str, image_type: str) -> Dict:
        """
        Analyze a medical image (X-ray, MRI, CT scan, etc.)
        Note: This is a mock implementation that would use ML models in a real system
        """
        # Mock image analysis based on image type
        analysis_result = {
            "image_type": image_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "findings": [],
            "confidence": 0.0,
            "recommendations": []
        }
        
        # Mock different analyses based on image type
        if image_type.lower() == "x-ray":
            # Simulate chest X-ray analysis
            analysis_result["findings"] = ["No significant abnormalities detected", 
                                           "Lung fields are clear", 
                                           "Heart size within normal limits"]
            analysis_result["confidence"] = 0.92
            analysis_result["recommendations"] = ["No further imaging required at this time"]
            
        elif image_type.lower() == "mri":
            # Simulate brain MRI analysis
            analysis_result["findings"] = ["No acute intracranial abnormality", 
                                           "No mass effect or midline shift", 
                                           "Ventricles normal in size"]
            analysis_result["confidence"] = 0.89
            analysis_result["recommendations"] = ["Clinical correlation recommended"]
            
        elif image_type.lower() == "ct":
            # Simulate abdominal CT analysis
            analysis_result["findings"] = ["Liver, spleen, and pancreas appear normal", 
                                           "No free fluid in the abdomen", 
                                           "No lymphadenopathy"]
            analysis_result["confidence"] = 0.87
            analysis_result["recommendations"] = ["Follow-up in 6 months recommended"]
        
        return analysis_result
    
    def process_symptoms(self, symptoms: List[str], patient_data: Dict) -> Dict:
        """
        Process patient symptoms and data to suggest potential diagnoses
        Note: This is a simplified implementation for demonstration
        """
        potential_conditions = []
        matched_symptoms = []
        
        # Process each symptom
        for symptom in symptoms:
            symptom = symptom.lower()
            # Check if symptom is in our database
            if symptom in SYMPTOM_DATABASE:
                matched_symptoms.append(symptom)
                potential_conditions.extend(SYMPTOM_DATABASE[symptom])
        
        # Count occurrences of each condition
        condition_counts = {}
        for condition in potential_conditions:
            if condition in condition_counts:
                condition_counts[condition] += 1
            else:
                condition_counts[condition] = 1
        
        # Sort conditions by number of matching symptoms
        sorted_conditions = sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate confidence based on symptom overlap
        top_conditions = []
        for condition, count in sorted_conditions[:3]:  # Top 3 conditions
            confidence = count / len(matched_symptoms) if matched_symptoms else 0
            top_conditions.append({
                "condition": condition,
                "confidence": round(confidence * 100, 1),
                "matching_symptoms": count
            })
        
        return {
            "matched_symptoms": matched_symptoms,
            "potential_conditions": top_conditions,
            "recommendation": "Please consult with a healthcare professional for proper diagnosis."
        }


class HealthcareDatabase:
    """Handles data storage and retrieval using Google Sheets as a database - all data saved to Prescription_Log"""
    
    def __init__(self, service_account_file=SERVICE_ACCOUNT_FILE):
        self.service_account_file = service_account_file
        self.client = None
        self.initialize_connection()
    
    def initialize_connection(self):
        """Initialize connection to Google Sheets API"""
        try:
            creds = Credentials.from_service_account_file(
                self.service_account_file, scopes=SCOPES
            )
            self.client = gspread.authorize(creds)
            print("Successfully connected to Google Sheets API")
        except Exception as e:
            print(f"Error connecting to Google Sheets: {e}")
    
    def get_or_create_prescription_sheet(self) -> Any:
        """Get an existing Prescription_Log sheet or create a new one if it doesn't exist"""
        try:
            # Try to open existing sheet
            spreadsheet = self.client.open(PRESCRIPTION_SHEET)
            print(f"Found existing spreadsheet: {PRESCRIPTION_SHEET}")
            sheet = spreadsheet.sheet1
            
            return sheet, spreadsheet.id
        
        except gspread.exceptions.SpreadsheetNotFound:
            # Create a new spreadsheet
            print(f"Creating new spreadsheet: {PRESCRIPTION_SHEET}")
            spreadsheet = self.client.create(PRESCRIPTION_SHEET)
            sheet = spreadsheet.sheet1
            
            # Initialize with comprehensive headers that cover all our data types
            sheet.append_row([
                "Record ID", "Record Type", "Patient ID", "Patient Name", 
                "Timestamp", "Extracted Text", "Medication", "Dosage", 
                "Instructions", "Confidence", "Price", "Quantity",
                "Total Price", "Status", "Image Type", "Findings",
                "Recommendations", "Symptoms", "Potential Conditions"
            ])
            
            return sheet, spreadsheet.id
    
    def log_prescription(self, patient_id: str, patient_name: str, 
                        extracted_text: str, medication_details: Dict) -> str:
        """Log a processed prescription to the database"""
        sheet, _ = self.get_or_create_prescription_sheet()
        
        # Generate record ID
        record_id = f"RX-{uuid.uuid4().hex[:8].upper()}"
        
        # Prepare row data
        row_data = [
            record_id,
            "Prescription",
            patient_id,
            patient_name,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            extracted_text,
            medication_details["name"],
            medication_details["dosage"],
            medication_details["instructions"],
            f"{medication_details['confidence']:.2f}",
            f"${medication_details['price']:.2f}",
            "", # Quantity (empty for prescription log)
            "", # Total Price (empty for prescription log)
            "", # Status (empty for prescription log)
            "", # Image Type (empty for prescription log)
            "", # Findings (empty for prescription log)
            "", # Recommendations (empty for prescription log)
            "", # Symptoms (empty for prescription log)
            "", # Potential Conditions (empty for prescription log)
        ]
        
        # Add to sheet
        sheet.append_row(row_data)
        print(f"Logged prescription for {patient_name}")
        
        return record_id
    
    def create_order(self, patient_id: str, patient_name: str, 
                    medication_details: Dict, quantity: int = 30) -> str:
        """Create a new medication order in the same Prescription_Log sheet"""
        sheet, _ = self.get_or_create_prescription_sheet()
        
        # Generate order ID
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        
        # Calculate total price
        total_price = medication_details["price"] * quantity
        
        # Prepare row data
        row_data = [
            order_id,
            "Order",
            patient_id,
            patient_name,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "", # Extracted Text (empty for order)
            medication_details["name"],
            medication_details["dosage"],
            medication_details["instructions"],
            "", # Confidence (empty for order)
            f"${medication_details['price']:.2f}",
            quantity,
            f"${total_price:.2f}",
            "Pending",
            "", # Image Type (empty for order)
            "", # Findings (empty for order)
            "", # Recommendations (empty for order)
            "", # Symptoms (empty for order)
            "", # Potential Conditions (empty for order)
        ]
        
        # Add to sheet
        sheet.append_row(row_data)
        print(f"Created order {order_id} for {patient_name}")
        
        return order_id
    
    def log_diagnostic_result(self, patient_id: str, patient_name: str, 
                             diagnostic_type: str, findings: List[str], 
                             recommendations: List[str], confidence: float) -> str:
        """Log a diagnostic result to the same Prescription_Log sheet"""
        sheet, _ = self.get_or_create_prescription_sheet()
        
        # Generate diagnostic ID
        diagnostic_id = f"DIAG-{uuid.uuid4().hex[:8].upper()}"
        
        # Prepare row data
        row_data = [
            diagnostic_id,
            "Diagnostic",
            patient_id,
            patient_name,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "", # Extracted Text (empty for diagnostic)
            "", # Medication (empty for diagnostic)
            "", # Dosage (empty for diagnostic)
            "", # Instructions (empty for diagnostic)
            f"{confidence:.2f}",
            "", # Price (empty for diagnostic)
            "", # Quantity (empty for diagnostic)
            "", # Total Price (empty for diagnostic)
            "", # Status (empty for diagnostic)
            diagnostic_type,
            "; ".join(findings),
            "; ".join(recommendations),
            "", # Symptoms (empty for diagnostic)
            "", # Potential Conditions (empty for diagnostic)
        ]
        
        # Add to sheet
        sheet.append_row(row_data)
        print(f"Logged diagnostic result {diagnostic_id} for {patient_name}")
        
        return diagnostic_id
    
    def log_symptom_analysis(self, patient_id: str, patient_name: str,
                           symptoms: List[str], conditions: List[Dict]) -> str:
        """Log a symptom analysis result to the same Prescription_Log sheet"""
        sheet, _ = self.get_or_create_prescription_sheet()
        
        # Generate symptom analysis ID
        symptom_id = f"SYM-{uuid.uuid4().hex[:8].upper()}"
        
        # Format potential conditions
        conditions_text = "; ".join([f"{c['condition']} ({c['confidence']}%)" for c in conditions])
        
        # Prepare row data
        row_data = [
            symptom_id,
            "Symptom Analysis",
            patient_id,
            patient_name,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "", # Extracted Text (empty for symptom analysis)
            "", # Medication (empty for symptom analysis)
            "", # Dosage (empty for symptom analysis)
            "", # Instructions (empty for symptom analysis)
            "", # Confidence (empty for symptom analysis)
            "", # Price (empty for symptom analysis)
            "", # Quantity (empty for symptom analysis)
            "", # Total Price (empty for symptom analysis)
            "", # Status (empty for symptom analysis)
            "", # Image Type (empty for symptom analysis)
            "", # Findings (empty for symptom analysis)
            "Please consult with a healthcare professional for proper diagnosis.",
            "; ".join(symptoms),
            conditions_text
        ]
        
        # Add to sheet
        sheet.append_row(row_data)
        print(f"Logged symptom analysis {symptom_id} for {patient_name}")
        
        return symptom_id


class HealthcareAssistant:
    """Main class that integrates prescription processing and diagnostic analysis"""
    
    def __init__(self, tesseract_path=r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        """Initialize the healthcare assistant with all components"""
        self.prescription_processor = PrescriptionProcessor(tesseract_path)
        self.diagnostic_analyzer = DiagnosticAnalyzer()
        self.database = HealthcareDatabase()
    
    def process_prescription(self, patient_id: str, patient_name: str, 
                           image_path: str, is_url: bool = False) -> Dict:
        """Process a prescription image and create an order"""
        try:
            # Process the prescription image
            if is_url:
                extracted_text, med_name, medication_details = self.prescription_processor.process_image_url(image_path)
            else:
                extracted_text, med_name, medication_details = self.prescription_processor.process_image_file(image_path)
            
            # Log the prescription to database
            record_id = self.database.log_prescription(patient_id, patient_name, extracted_text, medication_details)
            
            # Create an order
            order_id = self.database.create_order(patient_id, patient_name, medication_details)
            
            return {
                "status": "success",
                "record_id": record_id,
                "order_id": order_id,
                "patient_id": patient_id,
                "patient_name": patient_name,
                "medication": med_name,
                "dosage": medication_details["dosage"],
                "instructions": medication_details["instructions"],
                "confidence": medication_details["confidence"],
                "price": medication_details["price"],
                "extracted_text": extracted_text
            }
        
        except Exception as e:
            print(f"Error processing prescription: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error",
                "message": str(e)
            }
    
    def analyze_medical_image(self, patient_id: str, patient_name: str, 
                            image_path: str, image_type: str) -> Dict:
        """Analyze a medical diagnostic image"""
        try:
            # Analyze the medical image
            analysis_result = self.diagnostic_analyzer.analyze_medical_image(image_path, image_type)
            
            # Log the diagnostic result
            diagnostic_id = self.database.log_diagnostic_result(
                patient_id, 
                patient_name, 
                image_type,
                analysis_result["findings"],
                analysis_result["recommendations"],
                analysis_result["confidence"]
            )
            
            # Return combined result
            return {
                "status": "success",
                "diagnostic_id": diagnostic_id,
                "patient_id": patient_id,
                "patient_name": patient_name,
                "image_type": image_type,
                "findings": analysis_result["findings"],
                "recommendations": analysis_result["recommendations"],
                "confidence": analysis_result["confidence"]
            }
        
        except Exception as e:
            print(f"Error analyzing medical image: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error",
                "message": str(e)
            }
    
    def process_symptoms(self, patient_id: str, patient_name: str, 
                       symptoms: List[str], patient_data: Dict = None) -> Dict:
        """Process patient symptoms and suggest potential diagnoses"""
        try:
            # Process symptoms
            if patient_data is None:
                patient_data = {}
            
            diagnostic_result = self.diagnostic_analyzer.process_symptoms(symptoms, patient_data)
            
            # Log symptom analysis to the database
            symptom_id = self.database.log_symptom_analysis(
                patient_id,
                patient_name,
                diagnostic_result["matched_symptoms"],
                diagnostic_result["potential_conditions"]
            )
            
            # Return the result
            return {
                "status": "success",
                "symptom_id": symptom_id,
                "patient_id": patient_id,
                "patient_name": patient_name,
                "matched_symptoms": diagnostic_result["matched_symptoms"],
                "potential_conditions": diagnostic_result["potential_conditions"],
                "recommendation": diagnostic_result["recommendation"]
            }
        
        except Exception as e:
            print(f"Error processing symptoms: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error",
                "message": str(e)
            }


def process_user_prescription():
    """Function to get user input and process a prescription image from a URL"""
    # Initialize the healthcare assistant
    assistant = HealthcareAssistant()
    
    # Get user input
    patient_id = input("Enter patient ID: ")
    patient_name = input("Enter patient name: ")
    image_url = input("Enter image URL (Google Drive format: https://drive.google.com/uc?id=YOUR_FILE_ID): ")
    
    # Process the prescription from the provided URL
    print("\nProcessing prescription image. This may take a moment...")
    prescription_result = assistant.process_prescription(
        patient_id=patient_id,
        patient_name=patient_name,
        image_path=image_url,
        is_url=True
    )
    
    # Print the result
    print("\nPrescription Processing Result:")
    print(json.dumps(prescription_result, indent=2))
    
    return prescription_result


def process_user_symptoms():
    """Function to get user input and process symptoms"""
    # Initialize the healthcare assistant
    assistant = HealthcareAssistant()
    
    # Get user input
    patient_id = input("Enter patient ID: ")
    patient_name = input("Enter patient name: ")
    symptoms_input = input("Enter symptoms (separated by commas): ")
    
    # Parse symptoms
    symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
    
    # Get additional patient data
    age = input("Enter patient age (or press Enter to skip): ")
    sex = input("Enter patient sex (male/female, or press Enter to skip): ")
    allergies_input = input("Enter allergies (separated by commas, or press Enter to skip): ")
    allergies = [a.strip() for a in allergies_input.split(",") if a.strip()]
    
    # Create patient data dictionary
    patient_data = {}
    if age:
        patient_data["age"] = int(age)
    if sex:
        patient_data["sex"] = sex
    if allergies:
        patient_data["allergies"] = allergies
    
    # Process the symptoms
    print("\nProcessing symptoms...")
    symptoms_result = assistant.process_symptoms(
        patient_id=patient_id,
        patient_name=patient_name,
        symptoms=symptoms,
        patient_data=patient_data
    )
    
    # Print the result
    print("\nSymptom Analysis Result:")
    print(json.dumps(symptoms_result, indent=2))
    
    return symptoms_result


def analyze_user_medical_image():
    """Function to get user input and analyze a medical image"""
    # Initialize the healthcare assistant
    assistant = HealthcareAssistant()
    
    # Get user input
    patient_id = input("Enter patient ID: ")
    patient_name = input("Enter patient name: ")
    image_path = input("Enter image path (local file): ")
    image_type = input("Enter image type (X-ray, MRI, CT): ")
    
    # Analyze the medical image
    print("\nAnalyzing medical image...")
    image_result = assistant.analyze_medical_image(
        patient_id=patient_id,
        patient_name=patient_name,
        image_path=image_path,
        image_type=image_type
    )
    
    # Print the result
    print("\nMedical Image Analysis Result:")
    print(json.dumps(image_result, indent=2))
    
    return image_result


def main_menu():
    """Main menu function to select which operation to perform"""
    while True:
        print("\n====== Healthcare Assistant System ======")
        print("1. Process Prescription from URL")
        print("2. Process Patient Symptoms")
        print("3. Analyze Medical Image (local file)")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ")
        
        if choice == "1":
            process_user_prescription()
        elif choice == "2":
            process_user_symptoms()
        elif choice == "3":
            analyze_user_medical_image()
        elif choice == "4":
            print("Exiting system. Goodbye!")
            break
        else:
            print("Invalid choice. Please select a valid option.")


# Entry point of the program
if __name__ == "__main__":
    main_menu()

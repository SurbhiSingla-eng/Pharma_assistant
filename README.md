# Pharma_assistant
# Create the requirements.txt file
cat > requirements.txt << 'EOF'
pytesseract
Pillow
opencv-python
numpy
requests
gspread
google-auth
EOF

# Create a .gitignore file
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# Credentials
service_account.json
*.pem
*.key

# Temporary files
*.log
.DS_Store
EOF

# Create a README.md file
cat > README.md << 'EOF'
# Healthcare Assistant System

A comprehensive healthcare management system for processing prescriptions, analyzing medical images, and diagnosing symptoms.

## Features

- Prescription image processing with text extraction
- Medication recognition and matching
- Medical image analysis (X-ray, MRI, CT scans)
- Symptom analysis and diagnostic suggestions
- Integrated with Google Sheets for data storage

## Requirements

- Python 3.7+
- Tesseract OCR
- Google Cloud service account (for database functionality)

## Installation

1. Clone this repository
2. Install dependencies:
pip install -r requirements.txt

3. Install Tesseract OCR
4. Set up a Google Cloud service account and place the `service_account.json` file in the project root

## Usage

Run the main application:
The application provides a text-based menu to:
1. Process prescriptions from a URL
2. Process patient symptoms
3. Analyze medical images

## Project Structure

- `src/`: Main source code
  - `processors/`: Prescription and image processing
  - `analyzers/`: Diagnostic analysis
  - `database/`: Data storage functionality
  - `utils/`: Helper functions and constants
  - `main.py`: Entry point
EOF

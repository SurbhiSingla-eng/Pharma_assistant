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

1. Clone this repository:
   ```sh
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
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



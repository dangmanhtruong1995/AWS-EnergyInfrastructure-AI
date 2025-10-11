import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from pdb import set_trace
import requests
import math
from pathlib import Path
from scipy.spatial.distance import cdist

import asyncio

import PyPDF2  # or use pdfplumber, pymupdf
from io import BytesIO


def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            # set_trace()
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None

def main():
	pass

if __name__ == '__main__':
	main()
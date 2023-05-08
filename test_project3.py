from project3 import normalize_corpus, extract_text_function, smart_city_slicker
import pandas as pd
import numpy as np
import pytest


text = ""
def test_extract_text_function():
    # Test with a valid PDF file
    text = extract_text_function('OH Canton.pdf')
    assert isinstance(text, str)
    assert len(text) > 0

def test_normalize_corpus():
    normalize_corpus(text)



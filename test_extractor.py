#!/usr/bin/env python3
"""Test Davidsonian extractor"""
from davidsonian_extraction import DavidsonianExtractor

ext = DavidsonianExtractor()
test_sentences = [
    'The cat sat on the mat.',
    'A boy played with a ball.',
    'She walked to the store.'
]

for sent in test_sentences:
    result = ext.extract(sent)
    print(f'Sentence: {sent}')
    print(f'Extracted: {result}')
    print()

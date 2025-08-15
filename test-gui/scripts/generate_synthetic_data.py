#!/usr/bin/env python3
"""
Synthetic data generation for LLM training.
Powered by SynthexAI - https://synthex.theaniketgiri.me

Usage:
    python scripts/generate_synthetic_data.py --type code --size 10000
    python scripts/generate_synthetic_data.py --type medical --size 5000
"""

import argparse
import os
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional

class SyntheticDataGenerator:
    """Generate synthetic data for LLM training."""
    
    def __init__(self):
        self.templates = {
            'code': self._get_code_templates(),
            'medical': self._get_medical_templates(),
            'news': self._get_news_templates(),
            'fiction': self._get_fiction_templates(),
            'technical': self._get_technical_templates()
        }
    
    def _get_code_templates(self) -> List[str]:
        """Get code generation templates."""
        return [
            "def {function_name}({params}):
    """{docstring}"""
    {implementation}
    return {return_value}",
            "class {class_name}:
    def __init__(self, {params}):
        {init_implementation}
    
    def {method_name}(self, {method_params}):
        {method_implementation}",
            "import {module}

{code_block}",
            "if {condition}:
    {true_block}
else:
    {false_block}",
            "for {variable} in {iterable}:
    {loop_body}",
            "try:
    {try_block}
except {exception} as {error_var}:
    {except_block}",
            "def {function_name}({params}):
    {implementation}
    if {condition}:
        return {return_value1}
    return {return_value2}"
        ]
    
    def _get_medical_templates(self) -> List[str]:
        """Get medical text templates."""
        return [
            "The patient presented with {symptoms}. Upon examination, {findings}. The diagnosis was {diagnosis}.",
            "Treatment for {condition} typically involves {treatment}. {explanation}.",
            "The {procedure} was performed successfully. {outcome}.",
            "Common side effects of {medication} include {side_effects}. {management}.",
            "The {test} results showed {results}. {interpretation}.",
            "Risk factors for {disease} include {risk_factors}. {prevention}.",
            "The {organ} is responsible for {function}. {anatomy}."
        ]
    
    def _get_news_templates(self) -> List[str]:
        """Get news article templates."""
        return [
            "{location} - {event} occurred today, {description}. {impact}.",
            "Officials announced {announcement} in response to {situation}. {details}.",
            "The {industry} sector reported {news}. {analysis}.",
            "Research shows that {finding}. {implications}.",
            "Experts believe that {prediction}. {reasoning}.",
            "The government approved {policy}. {effects}.",
            "Local residents expressed {reaction} to {event}. {quotes}."
        ]
    
    def _get_fiction_templates(self) -> List[str]:
        """Get fiction writing templates."""
        return [
            "The {character} walked through the {setting}, {description}. {action}.",
            "In the distance, {observation}. {character} felt {emotion}.",
            "The {object} glowed with {quality}. {character} reached out and {action}.",
            "Memories of {past_event} flooded {character}'s mind. {reflection}.",
            "The {weather} created an atmosphere of {mood}. {character} {action}.",
            "Through the {obstacle}, {character} could see {vision}. {reaction}.",
            "The {sound} echoed through the {location}. {character} {response}."
        ]
    
    def _get_technical_templates(self) -> List[str]:
        """Get technical documentation templates."""
        return [
            "The {system} architecture consists of {components}. {explanation}.",
            "To configure {feature}, follow these steps: {steps}. {notes}.",
            "The {algorithm} operates by {process}. {complexity}.",
            "When {condition}, the system will {behavior}. {implications}.",
            "The {protocol} ensures {guarantee}. {implementation}.",
            "Performance metrics show {results}. {analysis}.",
            "The {interface} provides {functionality}. {usage}."
        ]
    
    def _get_random_words(self, category: str, count: int) -> List[str]:
        """Get random words for a category."""
        word_lists = {
            'function_name': ['process_data', 'calculate_result', 'validate_input', 'transform_data', 'generate_report'],
            'class_name': ['DataProcessor', 'UserManager', 'FileHandler', 'NetworkClient', 'DatabaseConnection'],
            'params': ['data', 'config', 'options', 'parameters', 'settings'],
            'docstring': ['Process the input data', 'Calculate the final result', 'Validate user input', 'Transform data format'],
            'implementation': ['result = data * 2', 'return data.upper()', 'data.sort()', 'data.reverse()'],
            'return_value': ['result', 'data', 'True', 'False', 'None'],
            'symptoms': ['fever', 'headache', 'nausea', 'fatigue', 'pain'],
            'diagnosis': ['common cold', 'migraine', 'food poisoning', 'stress', 'injury'],
            'treatment': ['rest', 'medication', 'therapy', 'surgery', 'lifestyle changes'],
            'location': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney'],
            'event': ['accident', 'announcement', 'discovery', 'meeting', 'celebration'],
            'character': ['hero', 'villain', 'witness', 'detective', 'teacher'],
            'setting': ['forest', 'city', 'castle', 'beach', 'mountain'],
            'emotion': ['fear', 'joy', 'sadness', 'anger', 'surprise'],
            'system': ['database', 'network', 'application', 'server', 'client'],
            'feature': ['authentication', 'logging', 'caching', 'monitoring', 'backup']
        }
        
        words = word_lists.get(category, ['example', 'sample', 'test', 'demo', 'placeholder'])
        return random.choices(words, k=count)
    
    def _fill_template(self, template: str) -> str:
        """Fill a template with random values."""
        # Find all placeholders
        placeholders = re.findall(r'{(w+)}', template)
        
        # Replace each placeholder
        result = template
        for placeholder in placeholders:
            words = self._get_random_words(placeholder, 1)
            result = result.replace(f'{{{placeholder}}}', words[0])
        
        return result
    
    def generate_data(self, data_type: str, size: int) -> List[str]:
        """
        Generate synthetic data.
        
        Args:
            data_type: Type of data to generate
            size: Number of samples to generate
            
        Returns:
            List of generated texts
        """
        if data_type not in self.templates:
            raise ValueError(f"Unknown data type: {data_type}")
        
        templates = self.templates[data_type]
        generated_texts = []
        
        for _ in range(size):
            # Select random template
            template = random.choice(templates)
            
            # Fill template
            text = self._fill_template(template)
            
            # Add some variation
            if random.random() < 0.3:
                text += f" {random.choice(['Additionally,', 'Furthermore,', 'Moreover,', 'However,'])} {self._fill_template(random.choice(templates))}"
            
            generated_texts.append(text)
        
        return generated_texts
    
    def save_data(self, texts: List[str], output_path: str):
        """Save generated data to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n\n')

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for LLM training")
    parser.add_argument("--type", "-t", choices=["code", "medical", "news", "fiction", "technical"], 
                       required=True, help="Type of data to generate")
    parser.add_argument("--size", "-s", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", "-o", help="Output file path")
    
    args = parser.parse_args()
    
    # Set default output path
    if not args.output:
        args.output = f"data/raw/synthetic_{args.type}_{args.size}.txt"
    
    try:
        # Create generator
        generator = SyntheticDataGenerator()
        
        # Generate data
        print(f"Generating {args.size} {args.type} samples...")
        texts = generator.generate_data(args.type, args.size)
        
        # Save data
        generator.save_data(texts, args.output)
        
        print(f"Generated {len(texts)} samples")
        print(f"Data saved to: {args.output}")
        
        # Show sample
        print("\nSample generated text:")
        print(texts[0])
        
        return 0
    except Exception as e:
        print(f"Error generating data: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

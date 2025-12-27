"""
Preprocessing script for bAbI dataset.
Converts bAbI tasks into a format suitable for training the neural logic core.
"""
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple


def parse_babi_file(filepath: str) -> List[Dict]:
    """Parse a bAbI task file and extract stories with entities and facts."""
    stories = []
    current_story = {
        "facts": [],
        "questions": []
    }
    entity_mapping = {}
    next_entity_id = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            line_num_and_text = parts[0].split(' ', 1)
            line_num = int(line_num_and_text[0])
            
            # New story
            if line_num == 1 and current_story["facts"]:
                stories.append(current_story)
                current_story = {"facts": [], "questions": []}
                entity_mapping = {}
                next_entity_id = 0
            
            if len(parts) == 1:  # This is a fact/statement
                text = line_num_and_text[1]
                
                # Extract entities from the text
                words = text.split()
                entities_in_fact = []
                
                for word in words:
                    # Simple heuristic: capitalized words are entities/locations
                    if word[0].isupper() or word in ['bathroom', 'hallway', 'garden', 'office', 'bedroom', 'kitchen']:
                        if word not in entity_mapping:
                            entity_mapping[word] = next_entity_id
                            next_entity_id += 1
                        entities_in_fact.append(entity_mapping[word])
                
                current_story["facts"].append({
                    "line_num": line_num,
                    "text": text,
                    "entities": entities_in_fact,
                    "entity_names": {eid: name for name, eid in entity_mapping.items()}
                })
            
            elif len(parts) >= 2:  # This is a question
                text = line_num_and_text[1]
                answer = parts[1].strip()
                supporting_facts = list(map(int, parts[2].strip().split())) if len(parts) > 2 else []
                
                # Map answer to entity ID
                answer_id = entity_mapping.get(answer, -1)
                
                current_story["questions"].append({
                    "line_num": line_num,
                    "text": text,
                    "answer": answer,
                    "answer_id": answer_id,
                    "supporting_facts": supporting_facts
                })
    
    # Add last story
    if current_story["facts"]:
        stories.append(current_story)
    
    return stories


def convert_to_logical_format(stories: List[Dict]) -> List[Dict]:
    """Convert stories to logical propositions format."""
    logical_stories = []
    
    for story in stories:
        entity_names = {}
        if story["facts"]:
            entity_names = story["facts"][0]["entity_names"]
        
        # Convert facts to logical triples
        facts_logical = []
        for fact in story["facts"]:
            text = fact["text"].lower()
            
            # Extract subject, relation, object
            # Simple pattern matching for bAbI Task 1
            if "moved to" in text or "went to" in text or "travelled to" in text or "journeyed to" in text:
                words = text.replace("moved to", "|").replace("went to", "|").replace("travelled to", "|").replace("journeyed to", "|").split("|")
                subject = words[0].strip().split()[0].capitalize()
                obj = words[1].strip().rstrip('.').capitalize()
                
                if subject in entity_names.values() or obj in entity_names.values():
                    # Get entity IDs
                    subj_id = [k for k, v in entity_names.items() if v == subject][0] if subject in entity_names.values() else -1
                    obj_id = [k for k, v in entity_names.items() if v == obj][0] if obj in entity_names.values() else -1
                    
                    facts_logical.append({
                        "line_num": fact["line_num"],
                        "subject_id": subj_id,
                        "subject": subject,
                        "relation": "located_at",
                        "object_id": obj_id,
                        "object": obj,
                        "text": fact["text"]
                    })
        
        logical_story = {
            "entities": entity_names,
            "facts": facts_logical,
            "questions": story["questions"]
        }
        logical_stories.append(logical_story)
    
    return logical_stories


def main():
    # Parse training data for Task 1
    data_dir = Path("data/tasks_1-20_v1-2/en-10k")
    train_file = data_dir / "qa1_single-supporting-fact_train.txt"
    test_file = data_dir / "qa1_single-supporting-fact_test.txt"
    
    print(f"Parsing training data from {train_file}...")
    train_stories = parse_babi_file(str(train_file))
    print(f"Parsed {len(train_stories)} training stories")
    
    print(f"Parsing test data from {test_file}...")
    test_stories = parse_babi_file(str(test_file))
    print(f"Parsed {len(test_stories)} test stories")
    
    # Convert to logical format
    print("Converting to logical format...")
    train_logical = convert_to_logical_format(train_stories)
    test_logical = convert_to_logical_format(test_stories)
    
    # Save processed data
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "task1_train.json", 'w') as f:
        json.dump(train_logical, f, indent=2)
    
    with open(output_dir / "task1_test.json", 'w') as f:
        json.dump(test_logical, f, indent=2)
    
    print(f"Saved processed data to {output_dir}")
    print(f"Training examples: {len(train_logical)}")
    print(f"Test examples: {len(test_logical)}")
    
    # Print example
    if train_logical:
        print("\nExample processed story:")
        print(json.dumps(train_logical[0], indent=2))


if __name__ == "__main__":
    main()

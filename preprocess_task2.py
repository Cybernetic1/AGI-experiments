"""
Preprocessing script for bAbI Task 2: Two Supporting Facts.
This task requires combining information from multiple facts.
"""
import json
from pathlib import Path
from typing import List, Dict


def parse_babi_task2(filepath: str) -> List[Dict]:
    """Parse bAbI Task 2 file."""
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
                
                # Extract entities - people, locations, objects
                words = text.split()
                entities_in_fact = []
                
                for word in words:
                    # Locations and people (capitalized)
                    if word[0].isupper():
                        if word not in entity_mapping:
                            entity_mapping[word] = next_entity_id
                            next_entity_id += 1
                        entities_in_fact.append(entity_mapping[word])
                    # Common locations and objects
                    elif word in ['bathroom', 'hallway', 'garden', 'office', 'bedroom', 
                                  'kitchen', 'football', 'milk', 'apple']:
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
            
            # Parse different fact types
            fact_dict = {
                "line_num": fact["line_num"],
                "text": fact["text"],
                "raw_text": text
            }
            
            # Movement facts
            if any(phrase in text for phrase in ["moved to", "went to", "travelled to", 
                                                   "journeyed to", "went back to"]):
                for phrase in ["moved to", "went to", "travelled to", "journeyed to", "went back to"]:
                    if phrase in text:
                        parts = text.replace(phrase, "|").split("|")
                        subject = parts[0].strip().split()[0].capitalize()
                        location = parts[1].strip().rstrip('.').split()[0]
                        fact_dict.update({
                            "type": "movement",
                            "subject": subject,
                            "relation": "moved_to",
                            "location": location
                        })
                        break
            
            # Object manipulation facts
            elif "got the" in text or "picked up the" in text:
                if "got the" in text:
                    parts = text.split("got the")
                else:
                    parts = text.split("picked up the")
                subject = parts[0].strip().split()[0].capitalize()
                obj = parts[1].strip().rstrip('.').split()[0]
                fact_dict.update({
                    "type": "pickup",
                    "subject": subject,
                    "relation": "has",
                    "object": obj
                })
            
            elif "dropped the" in text or "left the" in text:
                if "dropped the" in text:
                    parts = text.split("dropped the")
                else:
                    parts = text.split("left the")
                subject = parts[0].strip().split()[0].capitalize()
                obj = parts[1].strip().rstrip('.').split()[0]
                fact_dict.update({
                    "type": "drop",
                    "subject": subject,
                    "relation": "dropped",
                    "object": obj
                })
            
            elif "took the" in text:
                parts = text.split("took the")
                subject = parts[0].strip().split()[0].capitalize()
                obj = parts[1].strip().rstrip('.').split()[0]
                fact_dict.update({
                    "type": "pickup",
                    "subject": subject,
                    "relation": "has",
                    "object": obj
                })
            
            facts_logical.append(fact_dict)
        
        logical_story = {
            "entities": entity_names,
            "facts": facts_logical,
            "questions": story["questions"]
        }
        logical_stories.append(logical_story)
    
    return logical_stories


def main():
    # Parse training data for Task 2
    data_dir = Path("data/tasks_1-20_v1-2/en-10k")
    train_file = data_dir / "qa2_two-supporting-facts_train.txt"
    test_file = data_dir / "qa2_two-supporting-facts_test.txt"
    
    print(f"Parsing Task 2 training data from {train_file}...")
    train_stories = parse_babi_task2(str(train_file))
    print(f"Parsed {len(train_stories)} training stories")
    
    print(f"Parsing Task 2 test data from {test_file}...")
    test_stories = parse_babi_task2(str(test_file))
    print(f"Parsed {len(test_stories)} test stories")
    
    # Convert to logical format
    print("Converting to logical format...")
    train_logical = convert_to_logical_format(train_stories)
    test_logical = convert_to_logical_format(test_stories)
    
    # Save processed data
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "task2_train.json", 'w') as f:
        json.dump(train_logical, f, indent=2)
    
    with open(output_dir / "task2_test.json", 'w') as f:
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

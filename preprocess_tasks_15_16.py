"""
Preprocessing for Task 15 (Basic Deduction) and Task 16 (Basic Induction)
These tasks require learning and applying logical rules.
"""
import json
from pathlib import Path
from typing import List, Dict


def parse_logic_task(filepath: str, task_type: str) -> List[Dict]:
    """Parse Task 15 or 16 files."""
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
            
            if len(parts) == 1:  # Fact
                text = line_num_and_text[1]
                
                # Parse fact types
                fact_dict = {
                    "line_num": line_num,
                    "text": text
                }
                
                text_lower = text.lower()
                
                # Task 15: Category rules and instances
                if " are afraid of " in text_lower or " is afraid of " in text_lower:
                    if " are afraid of " in text_lower:
                        parts = text.split(" are afraid of ")
                        subject = parts[0].strip()
                        obj = parts[1].strip().rstrip('.')
                        fact_dict.update({
                            "type": "rule",
                            "category": subject.lower(),
                            "relation": "afraid_of",
                            "target": obj.lower()
                        })
                    else:
                        parts = text.split(" is afraid of ")
                        subject = parts[0].strip()
                        obj = parts[1].strip().rstrip('.')
                        fact_dict.update({
                            "type": "instance_relation",
                            "entity": subject,
                            "relation": "afraid_of",
                            "target": obj
                        })
                
                elif " is a " in text_lower:
                    parts = text.split(" is a ")
                    entity = parts[0].strip()
                    category = parts[1].strip().rstrip('.').rstrip('s')  # Remove plural 's'
                    fact_dict.update({
                        "type": "instance",
                        "entity": entity,
                        "category": category.lower()
                    })
                
                # Task 16: Color properties
                elif " is green" in text_lower or " is white" in text_lower or \
                     " is gray" in text_lower or " is yellow" in text_lower:
                    for color in ["green", "white", "gray", "grey", "yellow"]:
                        if f" is {color}" in text_lower:
                            entity = text.split(" is ")[0].strip()
                            fact_dict.update({
                                "type": "property",
                                "entity": entity,
                                "property": "color",
                                "value": color
                            })
                            break
                
                current_story["facts"].append(fact_dict)
            
            elif len(parts) >= 2:  # Question
                text = line_num_and_text[1]
                answer = parts[1].strip()
                supporting_facts = list(map(int, parts[2].strip().split())) if len(parts) > 2 else []
                
                current_story["questions"].append({
                    "line_num": line_num,
                    "text": text,
                    "answer": answer,
                    "supporting_facts": supporting_facts
                })
    
    if current_story["facts"]:
        stories.append(current_story)
    
    return stories


def main():
    data_dir = Path("data/tasks_1-20_v1-2/en-10k")
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    # Task 15: Basic Deduction
    print("Processing Task 15: Basic Deduction...")
    train_file = data_dir / "qa15_basic-deduction_train.txt"
    test_file = data_dir / "qa15_basic-deduction_test.txt"
    
    train_15 = parse_logic_task(str(train_file), "deduction")
    test_15 = parse_logic_task(str(test_file), "deduction")
    
    with open(output_dir / "task15_train.json", 'w') as f:
        json.dump(train_15, f, indent=2)
    with open(output_dir / "task15_test.json", 'w') as f:
        json.dump(test_15, f, indent=2)
    
    print(f"Task 15 - Train: {len(train_15)}, Test: {len(test_15)}")
    
    # Task 16: Basic Induction
    print("\nProcessing Task 16: Basic Induction...")
    train_file = data_dir / "qa16_basic-induction_train.txt"
    test_file = data_dir / "qa16_basic-induction_test.txt"
    
    train_16 = parse_logic_task(str(train_file), "induction")
    test_16 = parse_logic_task(str(test_file), "induction")
    
    with open(output_dir / "task16_train.json", 'w') as f:
        json.dump(train_16, f, indent=2)
    with open(output_dir / "task16_test.json", 'w') as f:
        json.dump(test_16, f, indent=2)
    
    print(f"Task 16 - Train: {len(train_16)}, Test: {len(test_16)}")
    
    # Show examples
    print("\n=== Task 15 Example ===")
    if train_15:
        print(json.dumps(train_15[0], indent=2))
    
    print("\n=== Task 16 Example ===")
    if train_16:
        print(json.dumps(train_16[0], indent=2))


if __name__ == "__main__":
    main()

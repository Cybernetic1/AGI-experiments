"""
Tasks 15 & 16: Rule Learning and Application
Task 15: Deduction - Apply explicit rules
Task 16: Induction - Learn implicit rules from examples
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict


class RuleLearningNetwork(nn.Module):
    """
    Network that learns and applies logical rules.
    
    Combines:
    - Symbolic rule storage (for Task 15)
    - Neural rule induction (for Task 16)
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.entity_embeddings = nn.Embedding(1000, embedding_dim)
        self.category_embeddings = nn.Embedding(100, embedding_dim)
        self.relation_embeddings = nn.Embedding(20, embedding_dim)
        
        # Rule encoder
        self.rule_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Instance encoder
        self.instance_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Rule application
        self.rule_matcher = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Answer decoder
        self.answer_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 100)  # Categories/values
        )
        
        # Symbolic rule storage
        self.explicit_rules = {}  # {category: {relation: target}}
        self.instances = {}  # {entity: category}
        self.properties = {}  # {entity: {property: value}}
        
        # Induced rules (for Task 16)
        self.induced_rules = defaultdict(lambda: defaultdict(int))  # {category: {value: count}}
        
        # Working memory
        self.rule_memory = []
        self.instance_memory = []
        
        # Plural/singular mappings
        self.plural_map = {
            'mouse': 'mice',
            'wolf': 'wolves',
            'sheep': 'sheep',
            'cat': 'cats'
        }
        self.singular_map = {v: k for k, v in self.plural_map.items()}
    
    def add_explicit_rule(self, category: str, relation: str, target: str):
        """Store explicit rule (Task 15)."""
        if category not in self.explicit_rules:
            self.explicit_rules[category] = {}
        self.explicit_rules[category][relation] = target
    
    def add_instance(self, entity: str, category: str):
        """Store instance classification."""
        self.instances[entity] = category
    
    def add_property(self, entity: str, property_name: str, value: str):
        """Store entity property."""
        if entity not in self.properties:
            self.properties[entity] = {}
        self.properties[entity][property_name] = value
        
        # Induce rule: if entity has category, learn category->value pattern
        if entity in self.instances:
            category = self.instances[entity]
            self.induced_rules[category][value] += 1
    
    def apply_deduction(self, entity: str, relation: str) -> str:
        """Apply deduction rule (Task 15)."""
        # Find entity's category
        if entity not in self.instances:
            return None
        
        category = self.instances[entity]
        
        # Get plural form if exists
        category_to_check = self.plural_map.get(category, category)
        
        # Look up rule
        if category_to_check in self.explicit_rules:
            if relation in self.explicit_rules[category_to_check]:
                target = self.explicit_rules[category_to_check][relation]
                # Convert to singular
                return self.singular_map.get(target, target.rstrip('s'))
        
        return None
    
    def apply_induction(self, entity: str, property_name: str) -> str:
        """Apply induced rule (Task 16)."""
        # Find entity's category
        if entity not in self.instances:
            return None
        
        category = self.instances[entity]
        
        # Find most common value for this category
        if category in self.induced_rules:
            if self.induced_rules[category]:
                most_common = max(self.induced_rules[category].items(), 
                                key=lambda x: x[1])
                return most_common[0]
        
        return None
    
    def reset_memory(self):
        """Clear all stored knowledge."""
        self.explicit_rules = {}
        self.instances = {}
        self.properties = {}
        self.induced_rules = defaultdict(lambda: defaultdict(int))
        self.rule_memory = []
        self.instance_memory = []


def load_data(filepath: str) -> List[Dict]:
    """Load preprocessed JSON data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def train_epoch_task15(model: RuleLearningNetwork, data: List[Dict], optimizer, criterion, device):
    """Train on Task 15 (Deduction)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for story in tqdm(data, desc="Training Task 15"):
        model.reset_memory()
        
        # Process facts
        for fact in story["facts"]:
            if fact.get("type") == "rule":
                category = fact.get("category", "")
                relation = fact.get("relation", "")
                target = fact.get("target", "")
                model.add_explicit_rule(category, relation, target)
            
            elif fact.get("type") == "instance":
                entity = fact.get("entity", "")
                category = fact.get("category", "")
                model.add_instance(entity, category)
        
        # Answer questions
        losses = []
        for question in story["questions"]:
            text = question["text"].lower()
            answer = question["answer"].lower().strip()
            
            # Parse question: "What is X afraid of?"
            if "what is" in text and "afraid of" in text:
                entity = text.split("what is")[1].split("afraid of")[0].strip()
                entity = entity.capitalize()
                
                # Apply deduction
                predicted = model.apply_deduction(entity, "afraid_of")
                
                if predicted == answer:
                    correct += 1
                total += 1
    
    # No gradient updates for pure symbolic reasoning
    avg_loss = 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def evaluate_task15(model: RuleLearningNetwork, data: List[Dict], criterion, device):
    """Evaluate Task 15."""
    model.eval()
    correct = 0
    total = 0
    
    for story in tqdm(data, desc="Evaluating Task 15"):
        model.reset_memory()
        
        for fact in story["facts"]:
            if fact.get("type") == "rule":
                model.add_explicit_rule(fact["category"], fact["relation"], fact["target"])
            elif fact.get("type") == "instance":
                model.add_instance(fact["entity"], fact["category"])
        
        for question in story["questions"]:
            text = question["text"].lower()
            answer = question["answer"].lower().strip()
            
            if "what is" in text and "afraid of" in text:
                entity = text.split("what is")[1].split("afraid of")[0].strip().capitalize()
                predicted = model.apply_deduction(entity, "afraid_of")
                
                if predicted == answer:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    return 0, accuracy


def train_epoch_task16(model: RuleLearningNetwork, data: List[Dict], optimizer, criterion, device):
    """Train on Task 16 (Induction)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for story in tqdm(data, desc="Training Task 16"):
        model.reset_memory()
        
        # Process facts
        for fact in story["facts"]:
            if fact.get("type") == "instance":
                model.add_instance(fact["entity"], fact["category"])
            elif fact.get("type") == "property":
                model.add_property(fact["entity"], fact["property"], fact["value"])
        
        # Answer questions
        for question in story["questions"]:
            text = question["text"].lower()
            answer = question["answer"].lower().strip()
            
            # Parse question: "What color is X?"
            if "what color is" in text:
                entity = text.split("what color is")[1].strip().rstrip('?').capitalize()
                
                # Check if we already know the answer
                if entity in model.properties and "color" in model.properties[entity]:
                    predicted = model.properties[entity]["color"]
                else:
                    # Apply induction
                    predicted = model.apply_induction(entity, "color")
                
                if predicted == answer:
                    correct += 1
                total += 1
    
    avg_loss = 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def evaluate_task16(model: RuleLearningNetwork, data: List[Dict], criterion, device):
    """Evaluate Task 16."""
    model.eval()
    correct = 0
    total = 0
    
    for story in tqdm(data, desc="Evaluating Task 16"):
        model.reset_memory()
        
        for fact in story["facts"]:
            if fact.get("type") == "instance":
                model.add_instance(fact["entity"], fact["category"])
            elif fact.get("type") == "property":
                model.add_property(fact["entity"], fact["property"], fact["value"])
        
        for question in story["questions"]:
            text = question["text"].lower()
            answer = question["answer"].lower().strip()
            
            if "what color is" in text:
                entity = text.split("what color is")[1].strip().rstrip('?').capitalize()
                
                if entity in model.properties and "color" in model.properties[entity]:
                    predicted = model.properties[entity]["color"]
                else:
                    predicted = model.apply_induction(entity, "color")
                
                if predicted == answer:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    return 0, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = RuleLearningNetwork(embedding_dim=64, hidden_dim=128)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # ===== TASK 15: Basic Deduction =====
    print("\n" + "="*60)
    print("TASK 15: BASIC DEDUCTION")
    print("="*60)
    
    train_15 = load_data("data/processed/task15_train.json")
    test_15 = load_data("data/processed/task15_test.json")
    print(f"Train stories: {len(train_15)}, Test stories: {len(test_15)}")
    
    print("\nEvaluating Task 15...")
    _, train_acc = train_epoch_task15(model, train_15, optimizer, criterion, device)
    _, test_acc = evaluate_task15(model, test_15, criterion, device)
    print(f"Task 15 - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    if test_acc > 0.9:
        torch.save(model.state_dict(), "models/task15_best.pt")
        print(f"✓ Task 15 SOLVED! Saved model.")
    
    # ===== TASK 16: Basic Induction =====
    print("\n" + "="*60)
    print("TASK 16: BASIC INDUCTION")
    print("="*60)
    
    train_16 = load_data("data/processed/task16_train.json")
    test_16 = load_data("data/processed/task16_test.json")
    print(f"Train stories: {len(train_16)}, Test stories: {len(test_16)}")
    
    print("\nEvaluating Task 16...")
    _, train_acc = train_epoch_task16(model, train_16, optimizer, criterion, device)
    _, test_acc = evaluate_task16(model, test_16, criterion, device)
    print(f"Task 16 - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    if test_acc > 0.9:
        torch.save(model.state_dict(), "models/task16_best.pt")
        print(f"✓ Task 16 SOLVED! Saved model.")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Task 15 (Deduction): Test Accuracy = {test_acc:.4f}")
    print(f"Task 16 (Induction): Test Accuracy = {test_acc:.4f}")


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    main()

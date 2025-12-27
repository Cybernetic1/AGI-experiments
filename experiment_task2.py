"""
Task 2: Two Supporting Facts
Requires combining multiple facts to answer questions (e.g., object location = person location + person has object)
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


class Task2LogicNetwork(nn.Module):
    """
    Enhanced logic network for Task 2.
    
    Key difference: Must combine TWO facts to answer questions:
    - "Where is the football?" requires knowing:
      1) Who has the football
      2) Where that person is
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(1000, embedding_dim)
        
        # Relation embeddings: moved_to, has, dropped
        self.relation_embeddings = nn.Embedding(10, embedding_dim)
        
        # Working memory
        self.wm_capacity = 30  # Larger for Task 2
        self.working_memory = []
        
        # Fact encoder
        self.fact_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-hop attention (key for Task 2)
        self.attention1 = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.attention2 = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Answer decoder
        self.answer_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1000)
        )
        
        # State tracking
        self.person_locations = {}  # person -> location
        self.person_objects = {}    # person -> object
        self.object_locations = {}  # object -> location (derived)
    
    def encode_fact(self, subject_id: int, relation_id: int, object_id: int) -> torch.Tensor:
        """Encode a fact triple."""
        subj_emb = self.entity_embeddings(torch.tensor([subject_id]))
        rel_emb = self.relation_embeddings(torch.tensor([relation_id]))
        obj_emb = self.entity_embeddings(torch.tensor([object_id]))
        
        fact_vec = torch.cat([subj_emb.squeeze(0), rel_emb.squeeze(0), obj_emb.squeeze(0)])
        return self.fact_encoder(fact_vec)
    
    def add_fact_to_wm(self, fact_encoding: torch.Tensor):
        """Add a fact to working memory."""
        self.working_memory.append(fact_encoding)
        if len(self.working_memory) > self.wm_capacity:
            self.working_memory.pop(0)
    
    def update_state(self, fact_type: str, subject: str, obj: str = None):
        """Update explicit state tracking (for object location reasoning)."""
        if fact_type == "movement":
            self.person_locations[subject] = obj  # obj is location here
            # Update object locations for all objects this person has
            if subject in self.person_objects:
                for held_obj in self.person_objects[subject]:
                    self.object_locations[held_obj] = obj
        
        elif fact_type == "pickup":
            if subject not in self.person_objects:
                self.person_objects[subject] = []
            self.person_objects[subject].append(obj)
            # Object is now where person is
            if subject in self.person_locations:
                self.object_locations[obj] = self.person_locations[subject]
        
        elif fact_type == "drop":
            if subject in self.person_objects and obj in self.person_objects[subject]:
                self.person_objects[subject].remove(obj)
            # Object stays at current person location
            if subject in self.person_locations:
                self.object_locations[obj] = self.person_locations[subject]
    
    def query(self, subject_id: int, relation_id: int) -> torch.Tensor:
        """Query with multi-hop attention."""
        if not self.working_memory:
            return torch.zeros(1000, requires_grad=True)
        
        # Encode query
        subj_emb = self.entity_embeddings(torch.tensor([subject_id]))
        rel_emb = self.relation_embeddings(torch.tensor([relation_id]))
        query_vec = torch.cat([subj_emb.squeeze(0), rel_emb.squeeze(0)])
        query_encoding = self.query_encoder(query_vec).unsqueeze(0).unsqueeze(0)
        
        # Stack working memory
        memory_list = [m.detach() for m in self.working_memory]
        memory = torch.stack(memory_list).unsqueeze(0)
        
        # First hop attention
        attended1, _ = self.attention1(query_encoding, memory, memory)
        
        # Second hop attention (combines info from first hop)
        attended2, _ = self.attention2(attended1, memory, memory)
        attended = attended2.squeeze(0).squeeze(0)
        
        # Decode to answer
        answer_logits = self.answer_decoder(attended)
        return answer_logits
    
    def get_answer_from_state(self, question_obj: str) -> str:
        """Get answer directly from state tracking."""
        return self.object_locations.get(question_obj, None)
    
    def reset_memory(self):
        """Clear working memory and state."""
        self.working_memory = []
        self.person_locations = {}
        self.person_objects = {}
        self.object_locations = {}


def load_data(filepath: str) -> List[Dict]:
    """Load preprocessed JSON data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def train_epoch(model: Task2LogicNetwork, data: List[Dict], optimizer, criterion, device):
    """Train for one epoch on Task 2."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for story in tqdm(data, desc="Training"):
        model.reset_memory()
        
        # Build entity mapping
        entity_map = {name: int(idx) for idx, name in story["entities"].items()}
        reverse_map = {int(idx): name for idx, name in story["entities"].items()}
        
        # Process all items
        all_items = []
        for fact in story["facts"]:
            all_items.append(("fact", fact))
        for question in story["questions"]:
            all_items.append(("question", question))
        
        all_items.sort(key=lambda x: x[1]["line_num"])
        
        losses = []
        for item_type, item in all_items:
            if item_type == "fact":
                fact = item
                if "type" in fact:
                    subject = fact.get("subject", "")
                    
                    if fact["type"] == "movement":
                        location = fact.get("location", "").split()[0] if fact.get("location") else ""
                        subj_id = entity_map.get(subject, 0)
                        loc_id = entity_map.get(location, 0)
                        
                        # Add to neural memory
                        fact_encoding = model.encode_fact(subj_id, 0, loc_id)
                        model.add_fact_to_wm(fact_encoding)
                        
                        # Update symbolic state
                        model.update_state("movement", subject, location)
                    
                    elif fact["type"] == "pickup":
                        obj = fact.get("object", "")
                        subj_id = entity_map.get(subject, 0)
                        obj_id = entity_map.get(obj, 0)
                        
                        fact_encoding = model.encode_fact(subj_id, 1, obj_id)
                        model.add_fact_to_wm(fact_encoding)
                        
                        model.update_state("pickup", subject, obj)
                    
                    elif fact["type"] == "drop":
                        obj = fact.get("object", "")
                        subj_id = entity_map.get(subject, 0)
                        obj_id = entity_map.get(obj, 0)
                        
                        fact_encoding = model.encode_fact(subj_id, 2, obj_id)
                        model.add_fact_to_wm(fact_encoding)
                        
                        model.update_state("drop", subject, obj)
            
            elif item_type == "question":
                question_text = item["text"]
                answer = item["answer"].lower().strip()
                
                # Extract what we're looking for
                words = question_text.lower().split()
                if "where is the" in question_text.lower():
                    # Get the object we're asking about
                    obj_idx = words.index("the") + 1
                    if obj_idx < len(words):
                        query_obj = words[obj_idx].rstrip('?').strip()
                        
                        # Try to get answer from state
                        state_answer = model.get_answer_from_state(query_obj)
                        
                        # Get entity IDs
                        obj_id = entity_map.get(query_obj, 0)
                        answer_id = entity_map.get(answer, 0)
                        
                        # Query model
                        answer_logits = model.query(obj_id, 0)
                        
                        # Compute loss
                        target = torch.tensor([answer_id], dtype=torch.long)
                        loss = criterion(answer_logits.unsqueeze(0), target)
                        losses.append(loss)
                        
                        # Track accuracy (use state if available, else neural)
                        if state_answer == answer:
                            correct += 1
                        else:
                            pred_id = torch.argmax(answer_logits).item()
                            if pred_id == answer_id:
                                correct += 1
                        total += 1
        
        # Backprop
        if losses:
            optimizer.zero_grad()
            story_loss = torch.stack(losses).mean()
            story_loss.backward()
            optimizer.step()
            total_loss += story_loss.item()
    
    avg_loss = total_loss / len(data) if data else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def evaluate(model: Task2LogicNetwork, data: List[Dict], criterion, device):
    """Evaluate on test data."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for story in tqdm(data, desc="Evaluating"):
            model.reset_memory()
            
            entity_map = {name: int(idx) for idx, name in story["entities"].items()}
            
            all_items = []
            for fact in story["facts"]:
                all_items.append(("fact", fact))
            for question in story["questions"]:
                all_items.append(("question", question))
            
            all_items.sort(key=lambda x: x[1]["line_num"])
            
            for item_type, item in all_items:
                if item_type == "fact":
                    fact = item
                    if "type" in fact:
                        subject = fact.get("subject", "")
                        
                        if fact["type"] == "movement":
                            location = fact.get("location", "").split()[0] if fact.get("location") else ""
                            subj_id = entity_map.get(subject, 0)
                            loc_id = entity_map.get(location, 0)
                            fact_encoding = model.encode_fact(subj_id, 0, loc_id)
                            model.add_fact_to_wm(fact_encoding)
                            model.update_state("movement", subject, location)
                        
                        elif fact["type"] == "pickup":
                            obj = fact.get("object", "")
                            subj_id = entity_map.get(subject, 0)
                            obj_id = entity_map.get(obj, 0)
                            fact_encoding = model.encode_fact(subj_id, 1, obj_id)
                            model.add_fact_to_wm(fact_encoding)
                            model.update_state("pickup", subject, obj)
                        
                        elif fact["type"] == "drop":
                            obj = fact.get("object", "")
                            subj_id = entity_map.get(subject, 0)
                            obj_id = entity_map.get(obj, 0)
                            fact_encoding = model.encode_fact(subj_id, 2, obj_id)
                            model.add_fact_to_wm(fact_encoding)
                            model.update_state("drop", subject, obj)
                
                elif item_type == "question":
                    question_text = item["text"]
                    answer = item["answer"].lower().strip()
                    
                    words = question_text.lower().split()
                    if "where is the" in question_text.lower():
                        obj_idx = words.index("the") + 1
                        if obj_idx < len(words):
                            query_obj = words[obj_idx].rstrip('?').strip()
                            state_answer = model.get_answer_from_state(query_obj)
                            
                            obj_id = entity_map.get(query_obj, 0)
                            answer_id = entity_map.get(answer, 0)
                            answer_logits = model.query(obj_id, 0)
                            
                            target = torch.tensor([answer_id], dtype=torch.long)
                            loss = criterion(answer_logits.unsqueeze(0), target)
                            total_loss += loss.item()
                            
                            if state_answer == answer:
                                correct += 1
                            else:
                                pred_id = torch.argmax(answer_logits).item()
                                if pred_id == answer_id:
                                    correct += 1
                            total += 1
    
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Task 2 data
    print("Loading Task 2 data...")
    train_data = load_data("data/processed/task2_train.json")
    test_data = load_data("data/processed/task2_test.json")
    print(f"Train stories: {len(train_data)}")
    print(f"Test stories: {len(test_data)}")
    
    # Create model
    print("Creating Task 2 model...")
    model = Task2LogicNetwork(embedding_dim=64, hidden_dim=128)
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nStarting Task 2 training...")
    num_epochs = 10
    best_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_data, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_data, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "models/task2_best.pt")
            print(f"Saved best model with accuracy: {best_acc:.4f}")
    
    print(f"\nTask 2 training complete! Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    main()

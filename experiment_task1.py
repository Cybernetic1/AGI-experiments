"""
First experiment: Train on bAbI Task 1 (Single Supporting Fact)
Simple baseline to validate the architecture works.
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm


class EntityRegistry:
    """Manages entity IDs and their embeddings."""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.entities = {}  # entity_name -> entity_id
        self.entity_embeddings = nn.Embedding(1000, embedding_dim)  # Max 1000 entities
        self.next_id = 0
    
    def get_or_create(self, entity_name: str) -> int:
        """Get existing entity ID or create new one."""
        if entity_name not in self.entities:
            self.entities[entity_name] = self.next_id
            self.next_id += 1
        return self.entities[entity_name]
    
    def get_embedding(self, entity_id: int) -> torch.Tensor:
        """Get embedding for an entity."""
        return self.entity_embeddings(torch.tensor(entity_id))


class SimpleLogicNetwork(nn.Module):
    """
    Simplified neural logic network for bAbI Task 1.
    
    This is a baseline that:
    1. Encodes facts as (subject, relation, object) triples
    2. Maintains working memory of recent facts
    3. Answers queries by retrieving relevant facts
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(1000, embedding_dim)
        
        # Relation embeddings (for now, just "located_at")
        self.relation_embeddings = nn.Embedding(10, embedding_dim)
        
        # Working memory: stores recent facts
        self.wm_capacity = 20
        self.working_memory = []
        
        # Fact encoder: encodes (subject, relation, object) -> hidden
        self.fact_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Query encoder: encodes (subject, relation, ?) -> hidden
        self.query_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism for memory retrieval
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Answer decoder: predicts object entity
        self.answer_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1000)  # Output logits over all possible entities
        )
    
    def encode_fact(self, subject_id: int, relation_id: int, object_id: int) -> torch.Tensor:
        """Encode a fact triple."""
        subj_emb = self.entity_embeddings(torch.tensor([subject_id]))
        rel_emb = self.relation_embeddings(torch.tensor([relation_id]))
        obj_emb = self.entity_embeddings(torch.tensor([object_id]))
        
        # Concatenate and encode
        fact_vec = torch.cat([subj_emb.squeeze(0), rel_emb.squeeze(0), obj_emb.squeeze(0)])
        return self.fact_encoder(fact_vec)
    
    def add_fact_to_wm(self, fact_encoding: torch.Tensor):
        """Add a fact to working memory."""
        self.working_memory.append(fact_encoding)
        if len(self.working_memory) > self.wm_capacity:
            self.working_memory.pop(0)
    
    def query(self, subject_id: int, relation_id: int) -> torch.Tensor:
        """Query: (subject, relation, ?) -> predict object."""
        if not self.working_memory:
            # No facts in memory, return random
            return torch.zeros(1000, requires_grad=True)
        
        # Encode query
        subj_emb = self.entity_embeddings(torch.tensor([subject_id]))
        rel_emb = self.relation_embeddings(torch.tensor([relation_id]))
        query_vec = torch.cat([subj_emb.squeeze(0), rel_emb.squeeze(0)])
        query_encoding = self.query_encoder(query_vec).unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
        
        # Stack working memory - create new tensor to ensure gradient flow
        memory_list = [m.detach() for m in self.working_memory]
        memory = torch.stack(memory_list).unsqueeze(0)  # (1, wm_size, hidden_dim)
        
        # Attention retrieval
        attended, _ = self.attention(query_encoding, memory, memory)
        attended = attended.squeeze(0).squeeze(0)  # (hidden_dim,)
        
        # Decode to answer
        answer_logits = self.answer_decoder(attended)
        return answer_logits
    
    def reset_memory(self):
        """Clear working memory (for new story)."""
        self.working_memory = []


def load_data(filepath: str) -> List[Dict]:
    """Load preprocessed JSON data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def train_epoch(model: SimpleLogicNetwork, data: List[Dict], optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for story in tqdm(data, desc="Training"):
        model.reset_memory()
        
        # Build entity mapping for this story
        entity_map = {name: int(idx) for idx, name in story["entities"].items()}
        
        # Process facts in order
        fact_idx = 0
        question_idx = 0
        
        # We need to interleave facts and questions based on line numbers
        all_items = []
        for fact in story["facts"]:
            all_items.append(("fact", fact))
        for question in story["questions"]:
            all_items.append(("question", question))
        
        # Sort by line number
        all_items.sort(key=lambda x: x[1]["line_num"])
        
        losses = []
        for item_type, item in all_items:
            if item_type == "fact":
                # Add fact to working memory
                subject = item["subject"]
                obj = item["object"].lower().strip()
                
                # Map to entity IDs
                subj_id = entity_map.get(subject, 0)
                obj_id = entity_map.get(obj, 0)
                
                # Encode and add to memory (relation_id = 0 for "located_at")
                fact_encoding = model.encode_fact(subj_id, 0, obj_id)
                model.add_fact_to_wm(fact_encoding)
            
            elif item_type == "question":
                # Answer question
                question_text = item["text"]
                answer = item["answer"].lower().strip()
                
                # Extract subject from question (simple heuristic)
                words = question_text.split()
                if len(words) >= 3:
                    subject = words[2].rstrip('?').strip()
                    
                    # Get entity IDs
                    subj_id = entity_map.get(subject, 0)
                    answer_id = entity_map.get(answer, 0)
                    
                    # Query model
                    answer_logits = model.query(subj_id, 0)
                    
                    # Compute loss
                    target = torch.tensor([answer_id], dtype=torch.long)
                    loss = criterion(answer_logits.unsqueeze(0), target)
                    losses.append(loss)
                    
                    # Track accuracy
                    pred_id = torch.argmax(answer_logits).item()
                    if pred_id == answer_id:
                        correct += 1
                    total += 1
        
        # Backprop accumulated losses for this story
        if losses:
            optimizer.zero_grad()
            story_loss = torch.stack(losses).mean()
            story_loss.backward()
            optimizer.step()
            total_loss += story_loss.item()
    
    avg_loss = total_loss / len(data) if data else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def evaluate(model: SimpleLogicNetwork, data: List[Dict], criterion, device):
    """Evaluate on test data."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for story in tqdm(data, desc="Evaluating"):
            model.reset_memory()
            
            # Build entity mapping
            entity_map = {name: int(idx) for idx, name in story["entities"].items()}
            
            # Process all items
            all_items = []
            for fact in story["facts"]:
                all_items.append(("fact", fact))
            for question in story["questions"]:
                all_items.append(("question", question))
            
            all_items.sort(key=lambda x: x[1]["line_num"])
            
            for item_type, item in all_items:
                if item_type == "fact":
                    subject = item["subject"]
                    obj = item["object"].lower().strip()
                    subj_id = entity_map.get(subject, 0)
                    obj_id = entity_map.get(obj, 0)
                    fact_encoding = model.encode_fact(subj_id, 0, obj_id)
                    model.add_fact_to_wm(fact_encoding)
                
                elif item_type == "question":
                    question_text = item["text"]
                    answer = item["answer"].lower().strip()
                    words = question_text.split()
                    if len(words) >= 3:
                        subject = words[2].rstrip('?').strip()
                        subj_id = entity_map.get(subject, 0)
                        answer_id = entity_map.get(answer, 0)
                        answer_logits = model.query(subj_id, 0)
                        target = torch.tensor([answer_id], dtype=torch.long)
                        loss = criterion(answer_logits.unsqueeze(0), target)
                        total_loss += loss.item()
                        pred_id = torch.argmax(answer_logits).item()
                        if pred_id == answer_id:
                            correct += 1
                        total += 1
    
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_data = load_data("data/processed/task1_train.json")
    test_data = load_data("data/processed/task1_test.json")
    print(f"Train stories: {len(train_data)}")
    print(f"Test stories: {len(test_data)}")
    
    # Create model
    print("Creating model...")
    model = SimpleLogicNetwork(embedding_dim=64, hidden_dim=128)
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nStarting training...")
    num_epochs = 10
    best_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_data[:100], optimizer, criterion, device)  # Start with subset
        test_loss, test_acc = evaluate(model, test_data[:20], criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "models/task1_best.pt")
            print(f"Saved best model with accuracy: {best_acc:.4f}")
    
    print(f"\nTraining complete! Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    main()

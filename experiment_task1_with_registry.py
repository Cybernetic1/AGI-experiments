"""
Task 1 with Persistent Entity Registry
Demonstrates how the global entity registry maintains knowledge across stories.
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from entity_registry import PersistentEntityRegistry


class Task1WithRegistry(nn.Module):
    """
    Enhanced Task 1 model using persistent entity registry.
    
    Key difference: Entities persist across stories with global IDs.
    """
    
    def __init__(self, entity_registry: PersistentEntityRegistry, 
                 embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.registry = entity_registry
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Use registry's embeddings
        self.entity_embeddings = entity_registry.entity_embeddings
        
        # Relation embeddings
        self.relation_embeddings = nn.Embedding(10, embedding_dim)
        
        # Working memory
        self.wm_capacity = 20
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
        
        # Attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Answer decoder - outputs entity IDs from registry
        self.answer_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10000)  # Match registry capacity
        )
    
    def encode_fact(self, subject_id: int, relation_id: int, object_id: int) -> torch.Tensor:
        """Encode a fact using registry entities."""
        subj_emb = self.entity_embeddings(torch.tensor([subject_id]))
        rel_emb = self.relation_embeddings(torch.tensor([relation_id]))
        obj_emb = self.entity_embeddings(torch.tensor([object_id]))
        
        fact_vec = torch.cat([subj_emb.squeeze(0), rel_emb.squeeze(0), obj_emb.squeeze(0)])
        return self.fact_encoder(fact_vec)
    
    def add_fact_to_wm(self, fact_encoding: torch.Tensor):
        """Add fact to working memory."""
        self.working_memory.append(fact_encoding)
        if len(self.working_memory) > self.wm_capacity:
            self.working_memory.pop(0)
    
    def query(self, subject_id: int, relation_id: int) -> torch.Tensor:
        """Query working memory."""
        if not self.working_memory:
            return torch.zeros(10000, requires_grad=True)
        
        subj_emb = self.entity_embeddings(torch.tensor([subject_id]))
        rel_emb = self.relation_embeddings(torch.tensor([relation_id]))
        query_vec = torch.cat([subj_emb.squeeze(0), rel_emb.squeeze(0)])
        query_encoding = self.query_encoder(query_vec).unsqueeze(0).unsqueeze(0)
        
        memory_list = [m.detach() for m in self.working_memory]
        memory = torch.stack(memory_list).unsqueeze(0)
        
        attended, _ = self.attention(query_encoding, memory, memory)
        attended = attended.squeeze(0).squeeze(0)
        
        answer_logits = self.answer_decoder(attended)
        return answer_logits
    
    def reset_memory(self):
        """Clear working memory (but registry persists!)."""
        self.working_memory = []


def load_data(filepath: str) -> List[Dict]:
    """Load preprocessed JSON data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def train_with_registry(model: Task1WithRegistry, data: List[Dict], 
                       optimizer, criterion, device):
    """Train using global entity registry."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    registry = model.registry
    
    for story in tqdm(data, desc="Training"):
        model.reset_memory()  # Clear WM, but entities persist in registry!
        
        # Map story entities to registry IDs
        local_to_global = {}  # {local_name: global_entity_id}
        
        # Register all entities from this story
        for local_id, name in story["entities"].items():
            # Determine type from name
            entity_type = None
            if name.lower() in ['bathroom', 'hallway', 'garden', 'office', 'bedroom', 'kitchen']:
                entity_type = 'location'
            elif name[0].isupper():
                entity_type = 'person'
            
            global_id = registry.get_or_create_entity(name, entity_type)
            local_to_global[name] = global_id
        
        # Process facts and questions
        all_items = []
        for fact in story["facts"]:
            all_items.append(("fact", fact))
        for question in story["questions"]:
            all_items.append(("question", question))
        
        all_items.sort(key=lambda x: x[1]["line_num"])
        
        losses = []
        for item_type, item in all_items:
            if item_type == "fact":
                subject = item.get("subject", "")
                obj = item.get("object", "").lower().strip()
                
                if subject in local_to_global and obj in local_to_global:
                    subj_id = local_to_global[subject]
                    obj_id = local_to_global[obj]
                    
                    # Store in registry
                    registry.add_relation(subj_id, "located_at", obj_id)
                    
                    # Add to working memory
                    fact_encoding = model.encode_fact(subj_id, 0, obj_id)
                    model.add_fact_to_wm(fact_encoding)
            
            elif item_type == "question":
                question_text = item["text"]
                answer = item["answer"].lower().strip()
                
                words = question_text.split()
                if len(words) >= 3:
                    subject = words[2].rstrip('?').strip()
                    
                    if subject in local_to_global and answer in local_to_global:
                        subj_id = local_to_global[subject]
                        answer_id = local_to_global[answer]
                        
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


def evaluate_with_registry(model: Task1WithRegistry, data: List[Dict], 
                          criterion, device):
    """Evaluate using global entity registry."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    registry = model.registry
    
    with torch.no_grad():
        for story in tqdm(data, desc="Evaluating"):
            model.reset_memory()
            
            local_to_global = {}
            for local_id, name in story["entities"].items():
                entity_type = None
                if name.lower() in ['bathroom', 'hallway', 'garden', 'office', 'bedroom', 'kitchen']:
                    entity_type = 'location'
                elif name[0].isupper():
                    entity_type = 'person'
                
                global_id = registry.get_or_create_entity(name, entity_type)
                local_to_global[name] = global_id
            
            all_items = []
            for fact in story["facts"]:
                all_items.append(("fact", fact))
            for question in story["questions"]:
                all_items.append(("question", question))
            
            all_items.sort(key=lambda x: x[1]["line_num"])
            
            for item_type, item in all_items:
                if item_type == "fact":
                    subject = item.get("subject", "")
                    obj = item.get("object", "").lower().strip()
                    
                    if subject in local_to_global and obj in local_to_global:
                        subj_id = local_to_global[subject]
                        obj_id = local_to_global[obj]
                        registry.add_relation(subj_id, "located_at", obj_id)
                        fact_encoding = model.encode_fact(subj_id, 0, obj_id)
                        model.add_fact_to_wm(fact_encoding)
                
                elif item_type == "question":
                    question_text = item["text"]
                    answer = item["answer"].lower().strip()
                    words = question_text.split()
                    
                    if len(words) >= 3:
                        subject = words[2].rstrip('?').strip()
                        
                        if subject in local_to_global and answer in local_to_global:
                            subj_id = local_to_global[subject]
                            answer_id = local_to_global[answer]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create persistent entity registry
    print("Creating persistent entity registry...")
    registry = PersistentEntityRegistry(embedding_dim=64)
    
    # Load data
    print("Loading data...")
    train_data = load_data("data/processed/task1_train.json")
    test_data = load_data("data/processed/task1_test.json")
    print(f"Train stories: {len(train_data)}, Test stories: {len(test_data)}")
    
    # Create model with registry
    print("Creating model with entity registry...")
    model = Task1WithRegistry(registry, embedding_dim=64, hidden_dim=128)
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nStarting training with persistent entity registry...")
    num_epochs = 10
    best_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_with_registry(model, train_data, optimizer, criterion, device)
        test_loss, test_acc = evaluate_with_registry(model, test_data, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Show registry stats
        stats = registry.get_stats()
        print(f"Registry: {stats['total_entities']} entities, "
              f"{stats['total_creations']} created, "
              f"{stats['total_lookups']} lookups")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "models/task1_registry_best.pt")
            registry.save("models/task1_registry.json")
            print(f"✓ Saved model and registry (accuracy: {best_acc:.4f})")
    
    print(f"\nTraining complete! Best test accuracy: {best_acc:.4f}")
    
    # Final registry stats
    print("\n=== Final Entity Registry Stats ===")
    stats = registry.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✓ Registry saved to models/task1_registry.json")


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    main()

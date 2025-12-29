"""
Preprocess TinyStories to extract logical propositions.

Uses spaCy for:
- Named entity recognition
- Dependency parsing
- Relation extraction

Converts natural language stories to structured propositions
compatible with our VQ-VAE + AR pipeline.
"""
import spacy
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
from collections import defaultdict


def load_stories(filepath: str) -> List[str]:
    """Load stories from file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    stories = content.split("---STORY_SEPARATOR---")
    stories = [s.strip() for s in stories if s.strip()]
    return stories


def extract_entities_from_story(doc, entity_counter: dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Extract entities from spaCy doc.
    
    Returns:
        local_to_global: {entity_name: global_id}
        global_to_local: {global_id: entity_name}
    """
    local_to_global = {}
    
    # Extract named entities (PERSON, ORG, GPE, etc.)
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'FAC', 'PRODUCT']:
            name = ent.text
            if name not in local_to_global:
                if name not in entity_counter:
                    entity_counter[name] = len(entity_counter)
                local_to_global[name] = entity_counter[name]
    
    # Extract important nouns (potential entities)
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']:
            # Get full noun phrase
            if token.dep_ in ['nsubj', 'nsubjpass', 'dobj', 'pobj']:
                # Skip pronouns and very common words
                if token.lower_ in ['it', 'he', 'she', 'they', 'i', 'you', 'we', 
                                   'something', 'someone', 'thing', 'things', 'one']:
                    continue
                
                name = token.text
                if name not in local_to_global:
                    if name not in entity_counter:
                        entity_counter[name] = len(entity_counter)
                    local_to_global[name] = entity_counter[name]
    
    global_to_local = {v: k for k, v in local_to_global.items()}
    return local_to_global, global_to_local


def extract_simple_propositions(doc, local_to_global: Dict[str, int]) -> List[Dict]:
    """
    Extract simple subject-verb-object propositions.
    
    Focus on simple patterns:
    - X is Y (copula)
    - X has Y (possession)
    - X goes to Y (motion)
    - X sees Y (perception)
    - X makes Y (creation)
    """
    propositions = []
    
    for sent in doc.sents:
        for token in sent:
            # Look for verb roots
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                # Find subject
                subject = None
                for child in token.children:
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        if child.text in local_to_global:
                            subject = child.text
                        break
                
                if not subject:
                    continue
                
                # Find object/complement
                obj = None
                relation = None
                
                # Check for copula (is/was)
                if token.lemma_ == 'be':
                    for child in token.children:
                        if child.dep_ in ['attr', 'acomp']:
                            if child.text in local_to_global:
                                obj = child.text
                                relation = 'is_a'
                            break
                
                # Check for direct object
                elif not obj:
                    for child in token.children:
                        if child.dep_ in ['dobj', 'pobj']:
                            if child.text in local_to_global:
                                obj = child.text
                            
                            # Determine relation type
                            verb_lemma = token.lemma_
                            if verb_lemma in ['have', 'has', 'had', 'own']:
                                relation = 'has'
                            elif verb_lemma in ['go', 'move', 'walk', 'run', 'come']:
                                relation = 'goes_to'
                            elif verb_lemma in ['see', 'find', 'meet', 'spot']:
                                relation = 'sees'
                            elif verb_lemma in ['make', 'build', 'create']:
                                relation = 'makes'
                            elif verb_lemma in ['like', 'love', 'want']:
                                relation = 'likes'
                            elif verb_lemma in ['give', 'share']:
                                relation = 'gives'
                            else:
                                relation = 'interacts_with'
                            break
                
                # Add proposition if we found both subject and object
                if subject and obj and relation:
                    propositions.append({
                        'subject': subject,
                        'subject_id': local_to_global[subject],
                        'relation': relation,
                        'object': obj,
                        'object_id': local_to_global[obj],
                        'text': sent.text
                    })
    
    return propositions


def preprocess_tinystories(num_stories: int = 1000, max_entities_per_story: int = 50):
    """
    Preprocess TinyStories into logical propositions.
    
    Args:
        num_stories: Number of stories to process
        max_entities_per_story: Skip stories with too many entities
    """
    print("=" * 60)
    print("Preprocessing TinyStories")
    print("=" * 60)
    
    # Load spaCy
    print("\nLoading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    print("✓ spaCy loaded")
    
    # Load stories
    print(f"\nLoading stories...")
    stories_file = "data/tinystories/stories_10000.txt"
    all_stories = load_stories(stories_file)
    print(f"✓ Loaded {len(all_stories)} stories")
    
    # Limit to num_stories
    stories_to_process = all_stories[:num_stories]
    print(f"\nProcessing {len(stories_to_process)} stories...")
    
    # Global entity counter
    entity_counter = {}
    processed_stories = []
    skipped = 0
    
    for story_text in tqdm(stories_to_process, desc="Processing"):
        # Parse with spaCy
        doc = nlp(story_text)
        
        # Extract entities
        local_to_global, global_to_local = extract_entities_from_story(doc, entity_counter)
        
        # Skip stories with too many entities (likely parsing errors)
        if len(local_to_global) > max_entities_per_story or len(local_to_global) < 2:
            skipped += 1
            continue
        
        # Extract propositions
        propositions = extract_simple_propositions(doc, local_to_global)
        
        # Skip stories with no propositions
        if not propositions:
            skipped += 1
            continue
        
        # Create story dict
        processed_story = {
            'text': story_text,
            'entities': global_to_local,
            'facts': propositions
        }
        processed_stories.append(processed_story)
    
    print(f"\n✓ Processed {len(processed_stories)} stories")
    print(f"  Skipped: {skipped} (too few/many entities or no propositions)")
    print(f"  Total unique entities: {len(entity_counter)}")
    
    # Statistics
    total_propositions = sum(len(s['facts']) for s in processed_stories)
    avg_entities = sum(len(s['entities']) for s in processed_stories) / len(processed_stories)
    avg_props = total_propositions / len(processed_stories)
    
    print(f"\nStatistics:")
    print(f"  Total propositions: {total_propositions}")
    print(f"  Avg entities per story: {avg_entities:.1f}")
    print(f"  Avg propositions per story: {avg_props:.1f}")
    
    # Analyze relation types
    relation_counts = defaultdict(int)
    for story in processed_stories:
        for fact in story['facts']:
            relation_counts[fact['relation']] += 1
    
    print(f"\nRelation types:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel}: {count} ({count/total_propositions*100:.1f}%)")
    
    # Split train/test (90/10)
    split_idx = int(len(processed_stories) * 0.9)
    train_stories = processed_stories[:split_idx]
    test_stories = processed_stories[split_idx:]
    
    print(f"\nTrain/test split:")
    print(f"  Train: {len(train_stories)} stories")
    print(f"  Test: {len(test_stories)} stories")
    
    # Save
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "tinystories_train.json", 'w') as f:
        json.dump(train_stories, f, indent=2)
    
    with open(output_dir / "tinystories_test.json", 'w') as f:
        json.dump(test_stories, f, indent=2)
    
    # Save entity index
    with open(output_dir / "tinystories_entities.json", 'w') as f:
        json.dump(entity_counter, f, indent=2)
    
    print(f"\n✓ Saved to {output_dir}")
    
    # Show example
    print("\n" + "=" * 60)
    print("Example processed story:")
    print("=" * 60)
    example = processed_stories[0]
    print(f"\nOriginal text (first 200 chars):")
    print(example['text'][:200] + "...")
    print(f"\nEntities: {example['entities']}")
    print(f"\nPropositions:")
    for fact in example['facts'][:5]:
        print(f"  {fact['subject']} --{fact['relation']}--> {fact['object']}")
        print(f"    (from: {fact['text'][:60]}...)")
    
    return processed_stories


if __name__ == "__main__":
    # Process 1000 stories (manageable size for testing)
    processed = preprocess_tinystories(num_stories=1000)
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing complete!")
    print("\nNext: Train VQ-VAE and AR model on TinyStories")
    print("=" * 60)

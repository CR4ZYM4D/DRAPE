import json
import random
import os

# Base templates based on the 5 evaluation queries
# 1. "A person in a bright yellow raincoat." — attribute-specific
# 2. "Professional business attire inside a modern office." — contextual/place
# 3. "Someone wearing a blue shirt sitting on a park bench." — complex semantic
# 4. "Casual weekend outfit for a city walk." — style inference
# 5. "A red tie and a white shirt in a formal setting." — compositional

garments = ["raincoat", "shirt", "blouse", "t-shirt", "sweater", "jacket", "pants", "skirt", "coat", "dress", "tie", "hat", "scarf"]
colors = ["red", "blue", "green", "yellow", "black", "white", "grey", "brown", "beige", "pink", "purple"]
contexts = ["office", "park", "street", "home", "gym", "beach", "restaurant"]
styles = ["casual", "formal", "professional", "sporty", "elegant", "relaxed", "minimalist"]

def generate_queries(num=100):
    queries = []
    
    # Add the 5 originals
    originals = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting."
    ]
    queries.extend([{"id": i+1, "query": q, "type": "original"} for i, q in enumerate(originals)])
    
    for i in range(len(originals) + 1, num + 1):
        q_type = random.choice(["attribute-specific", "contextual", "semantic", "style", "compositional"])
        
        if q_type == "attribute-specific":
            g = random.choice(garments)
            c = random.choice(colors)
            q = f"A person wearing a {c} {g}."
        elif q_type == "contextual":
            s = random.choice(styles)
            cx = random.choice(contexts)
            q = f"{s.capitalize()} attire suitable for the {cx}."
        elif q_type == "semantic":
            g = random.choice(garments)
            c = random.choice(colors)
            cx = random.choice(contexts)
            q = f"Someone in a {c} {g} at the {cx}."
        elif q_type == "style":
            s = random.choice(styles)
            q = f"A very {s} outfit for today."
        elif q_type == "compositional":
            g1, g2 = random.sample(garments, 2)
            c1, c2 = random.sample(colors, 2)
            s = random.choice(styles)
            q = f"A {c1} {g1} paired with a {c2} {g2} in a {s} look."
            
        queries.append({"id": i, "query": q, "type": q_type})
        
    return queries

if __name__ == "__main__":
    queries = generate_queries(100)
    output_path = os.path.join(os.path.dirname(__file__), "sample_queries.json")
    with open(output_path, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"Generated 100 queries to {output_path}")

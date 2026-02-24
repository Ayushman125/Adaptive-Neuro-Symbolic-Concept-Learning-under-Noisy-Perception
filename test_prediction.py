"""
Test script to verify prediction logic works correctly.
Tests sci-fi concept learning and verifies non-movies are rejected.
"""

import subprocess
import sys

# Test inputs: positive examples (sci-fi), negative examples (non-sci-fi), then non-movies
test_inputs = [
    # Positive examples - sci-fi movies
    ("Dune, Blade Runner 2049, The Matrix, Interstellar", "y"),
    
    # Negative examples - non-sci-fi movies
    ("The Godfather, Titanic, Forrest Gump", "n"),
    
    # More negative examples
    ("Braveheart, Gladiator, The Shawshank Redemption", "n"),
    
    # Positive example - more sci-fi
    ("Star Wars, Inception, Ex Machina", "y"),
    
    # Test items that should predict NO-MATCH (not movies)
    ("paper", "n"),
    ("tree", "n"),
    ("camera", "n"),
    ("donkey", "n"),
    
    # Final sci-fi movies to confirm pattern still holds
    ("Arrival, District 9, Alien", "y"),
    
    # Exit
    ("", "")
]

print("=" * 70)
print("AUTOMATED TEST: Sci-Fi Concept Learning")
print("=" * 70)
print("\nTest Plan:")
print("1. Train on sci-fi movies (YES)")
print("2. Train on non-sci-fi movies (NO)")
print("3. Test with non-movies (paper, tree, camera, donkey)")
print("4. Expected: Should predict NO-MATCH for non-movies\n")

# Prepare input string
input_data = ""
for item, answer in test_inputs:
    if item:
        input_data += f"{item}\n{answer}\n"
        # Skip active learning questions by answering 'n' to them
        input_data += "n\n" * 3  # Answer 'n' to any active learning questions
    else:
        break

try:
    # Run the program with simulated input using venv Python
    python_exe = r"C:\Users\HP\views\venv\Scripts\python.exe"
    process = subprocess.Popen(
        [python_exe, "Thinkingmachiene.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=r"C:\Users\HP\views"
    )
    
    stdout, stderr = process.communicate(input=input_data, timeout=60)
    
    print("\n" + "=" * 70)
    print("TEST OUTPUT:")
    print("=" * 70)
    print(stdout)
    
    if stderr:
        print("\n" + "=" * 70)
        print("ERRORS:")
        print("=" * 70)
        print(stderr)
    
    # Analyze results
    print("\n" + "=" * 70)
    print("TEST ANALYSIS:")
    print("=" * 70)
    
    lines = stdout.split('\n')
    predictions = []
    
    for i, line in enumerate(lines):
        if "I predict this is a" in line and "UNCERTAIN" not in line:
            # Extract prediction
            if "NO-MATCH" in line:
                pred = "NO-MATCH"
            elif "MATCH" in line:
                pred = "MATCH"
            else:
                continue
            
            # Find the corresponding input item (look backwards)
            item = None
            for j in range(i-1, max(0, i-20), -1):
                if "[Input Item]:" in lines[j]:
                    item = lines[j].split("]:", 1)[1].strip()
                    break
            
            if item:
                predictions.append((item, pred))
    
    print("\nPredictions made:")
    for item, pred in predictions:
        item_short = item[:50] + "..." if len(item) > 50 else item
        print(f"  {pred:12s} - {item_short}")
    
    # Check critical test cases
    print("\n" + "=" * 70)
    print("CRITICAL CHECKS:")
    print("=" * 70)
    
    test_items = ["paper", "tree", "camera", "donkey"]
    passed = True
    
    for test_item in test_items:
        found = False
        for item, pred in predictions:
            if test_item in item.lower():
                found = True
                if pred == "NO-MATCH":
                    print(f"✓ PASS: '{test_item}' correctly predicted as NO-MATCH")
                else:
                    print(f"✗ FAIL: '{test_item}' incorrectly predicted as {pred}")
                    passed = False
                break
        
        if not found:
            print(f"? UNKNOWN: No prediction found for '{test_item}'")
    
    print("\n" + "=" * 70)
    if passed:
        print("✓ TEST PASSED: All non-movies correctly rejected!")
    else:
        print("✗ TEST FAILED: Some non-movies were incorrectly predicted as MATCH")
    print("=" * 70)
    
except subprocess.TimeoutExpired:
    print("\n✗ TEST TIMEOUT: Program took too long to respond")
    process.kill()
except Exception as e:
    print(f"\n✗ TEST ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

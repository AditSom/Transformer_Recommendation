import random


# traits
gender = ["male", "female"]
heights = ["5'8", "5'9", "5'10", "5'11", "6'0", "6'1", "6'2"]
hobbies = ["basketball", "football", "soccer", "hiking", "reading", "writing", "coding", "swimming", "running", "gaming"]
hobbies += ["cooking", "baking", "painting", "drawing", "dancing", "singing", "playing an instrument", "yoga", "meditation"]

drinking = ["socially", "rarely", "never", "often"]
smoking = ["socially", "rarely", "never", "often"]
looking_for = ["serious only", "somewhat serious", "something casual"]

traits = [gender, heights, hobbies, drinking, smoking, looking_for]

# Generate a random personality
personality = ["gender", "height", "hobbies", "drinking", "smoking", "looking for"]

for i, trait in enumerate(traits):
    if trait == hobbies:
        personality[i] += ": " + ", ".join(random.sample(trait, 5))
    else:
        personality[i] += ": " + random.choice(trait)

#store these personalities and match with images through random sampling
print(personality)
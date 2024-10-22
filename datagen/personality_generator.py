import random
from openai import OpenAI

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

print(personality)

key = "sk-kD601DP0S_xHBtQ9Dccrw5CNQgreh-5_yHta1Zfu89T3BlbkFJjZ5y7204kg90f0tw6-WCE6Jmx47wjkvT6JcHcsmMEA"

client = OpenAI(
    api_key = key,
)

stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
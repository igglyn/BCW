# Byte Conditioning Wave

That was the working name for it anyways, does it even count as that anymore?
The point of the Conditioning part was to draw a parallel to the conditioning you'd find in diffusion models
But supassing all expatations, it manages to keep quite a shocking amount of context.


# Data Requirements
As of writing this, decent results in constrained sizing requires around 1.5 Epochs by back of hand math.
Hasn't been explitly tested, but it does need to visit quite a few examples to reconize patterns,
with the general rule of more data for better results.

Wikitext is used for testing, as overfitting on a small dataset causes worse results.

```py
from pathlib import Path
import datasets

# 1. Load the dataset
# Wikitext is 100mb, hence why a script is proviced
dataset = datasets.load_dataset("wikitext", "wikitext-103-v1")

# 2. Define where to save the files (we split interally)
save_file = open("wikitext.txt", mode="w")

# 3. Iterate through each split (train, validation, test) and save to a text file
for split_name in ("train", "validation", "test"):
    print(f"Saving {split_name} split...")
    for example in dataset[split_name]:
        save_file.write(example['text'])

print(f"Done! Files saved in '{save_file}'")
```

# Tuning
There are a few tricks to this if you want to save time, the default config already has most of my recommendations built in.
But for transparency on why the choices were made.
- Patching doesn't require a large batch size as training a prediction model does. Generally you prefer steps over batch in a time crunch.
    - If you notice pratically every loss starting to wall, the first instinct might be to decrease steps for faster turn-around.
    But that time is important as well, it serves as the fine-tuning stage.
- d_model turns out can be pretty tight, like honestly to an insane degree. The patching can do it's job well, even in constrained spaces.
- Do not get fooled by the byte reconsruction being 99%, bytes are one thing and there are plenty of them in a span. All of the 1% chances add up.

# Background
I want to get this released ASAP, so I'll expand in a later commit

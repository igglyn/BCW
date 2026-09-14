# Byte Conditioning Wave

That was the working name for it anyways, does it even count as that anymore?
The point of the Conditioning part was to draw a parallel to the conditioning you'd find in diffusion models
But surpassing all expectations, it manages to keep quite a shocking amount of context.


# Data Requirements
As of writing this, decent results in constrained sizing requires around 1.5 Epochs by back of hand math.
Hasn't been explicitly tested, but it does need to visit quite a few examples to recognize patterns,
with the general rule of more data for better results.

Wikitext is used for testing, as overfitting on a small dataset causes worse results.

```py
from pathlib import Path
import datasets

# 1. Load the dataset
# Wikitext is 100mb, hence why a script is provided
dataset = datasets.load_dataset("wikitext", "wikitext-103-v1")

# 2. Define where to save the files (we split internally)
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
    - If you notice practically every loss starting to wall, the first instinct might be to decrease steps for faster turn-around.
    But that time is important as well, it serves as the fine-tuning stage.
- d_model turns out can be pretty tight, like honestly to an insane degree. The patching can do it's job well, even in constrained spaces.
- Do not get fooled by the byte reconstruction being 99%, bytes are one thing and there are plenty of them in a span. All of the 1% chances add up.
- If by 1k steps it seems like it is not going to work, it isn't going to and your asking to compress too much.

# Background
The origins of this follows a decently well chain of events:
```
> Patching using a small model, using Byte Latent Transformers
> 6gb 3060
> I'm forced to learn the scaling laws of this approach
> Looks like that it works well with more steps rather than raw batch exposure
> d_model 16, 16 batch, 16 tokens with RoPE; Noted
> Will it stack?
> yes, it can stack another layer
> two layers can get close to 128 bytes, (16*8)
> Converges but start to not really like transformers
> Slow, takes forever
> Several months go by, project retired
> Picking back up the slack due to intrusive thought
> Instead of using transformers again, let's use this Wave Network which I've been wanting to use
> Working well, it can match what I need it to
> Using 16 and 32 wide, pulling from the previous efforts
> The JEPA model I was using for prediction and also using Wave Network seems to not like predicting
> Tries a million ways to botch it into working
> Alright, if I can't be smart here then I have to assume elsewhere
> Let's make the prediction easier by less information to predict.
> Moving the second layer from 32 -> 16, works fine. 1:1 ratio
> "That's one reduction for JEPA, as that is 1:1 with the second layer"
> Intrusive thought: can the first one go lower
> Alright, sure let me just reduce then
> 16 -> 8
> Fine enough, would be funny to see 4 collapse
> 8 -> 4, doesn't collapse
> Whar?
> Alright, surely the 1:1 ratio from before doesn't work
> It totally does
> What is even going on
> "I'm tired of having to deal with both of these, the first one is still using space delimited instead of learned"
> Let's try and combine both of these into something greater than the sum of it's parts
> "Why not just make the model compress unconditionally"
> Alright sure, let's only have a single encoder; place both objectives on it and see if it works out.
> Because these should be able to make it flexible I need a flexible output that survives training
> Alright learned masking, as it learns to use less space it will compress down
> Let's run this, and second ratio is converging... First layer is also getting close
> Odd, I expected you to your limit
> Continue with a longer than 4k step training run
> Ratio2 has converged completely, Ratio1 is not slowing down
> Frick it cut off before it finished
> One more time
> Alright it has converged
> *realization*
> How is that converging hold on a minute, that is 256 bytes into one embedding
> Accuracy continues to rise
> Lot of testing later, from 75 to about 91% for chunk (aka exact match)
> Welp, that just happened.
> *The feeling sets in, what have you done*
> Let's recap, I have thrown 256 bytes into a single 4 wide embedding, in one layer, and this is only ~4k parameters
> Welp, I'm never recovering from the high that this has become.
> Let's make sure this is useful
> BLT2 training run
> NaN
> Odd, that was just working
> NaN 3 more times
> Okay, that looks like that is unstable...
> How did I even get the initial results that allowed me to yolo this
> Looks like BWC is a lot more stable than where it came from
> Sure the 2x parameter cost burns but in exchange for this thing being testable sure
> There is a lot more to do, but for now let's wrap this up and let it be
```

# Present
```
> High off of the success of the project
> There has to be similar places where this can go
> Command the slopper to research
> Alright, I guess geometric looks interesting
> How does one merge all of this into Wave
> Good question, let's start with just 2d
> Connected most of the points over to what Wave is doing
> minimal additions
> Offhandedly mention wavelet, which was already supposed to be here
> "there is no wavelet here"
> well frick I guess that is also being added this time
> Capacity constinues to grow, 32k
> looking good
> Next axis is sparsity
> One of yous really should be implmented outside of this repo
> add in the mechanisms in question
> That is getting up to 131k good
> I need to deal with these parameters
> decoder is over 500k large
> Surely this can be easily dealt with
> "This constant size encoder assumes relative positions only"
> I've had that constraint for ages
> Preforms much better
> That's that solved then
> Pretty small in size now
> Paranoia
> Let me see if this is actually compressing
> It is not
> https://github.com/igglyn/BCW/commit/96c42609cce03c8244c1e269217c40adc42f924b
> Welp that means every edit so far has only increasing the capacity for representation capacity
> Moving back down to sane starting values
> Alright it still compresses well
> moving back up
> Man these extra losses kinda suck now
> Atomized
> this LR is kinda balls, let's increase you
> MORE
> MORE
> 1%
> MOR- Nah that is unstable
> Alright that is much faster
> Completely invalidated by 131k taking just about 1M steps
> I still want more of these parameters gone
> Observe the encoder and decoder's mix values
> Decoder suppresses all urge to multiply
> Fine let's just fix that one so it never has to
> Encoder is similarly approaching multiplication
> One run gets nuked by it never reaches it
> This is a bugfix
> Okay that has changed the training to rather extreme degree
> "It no longer trades ratio to increase the representation cost"
> I am out of ideas
> Figuring I should just release, yap about future plans
> New idea, that 2d Wavelet; what about it becomes higher dim
> 3d manages to compress much better at slightly better results
> 4d holds decently well
> Alright that changes the framing decently
> Let's all in
> allergic to more steps
> "odd, you managed to finally how that behavior for a patching model"
> Increasing stride one last time then
```

# AI Usage Report

## Which AI tool I used

I used Claude (made by Anthropic) through Claude Code on the web for the whole project. I didn't use any other AI tools.

## How I used it

I'll be honest: most of the code in this repository was written by Claude, not typed from scratch by me. I have intermediate Python experience, but I had never built a full deep learning project before, and I had a tight deadline. So I used the AI more like a senior colleague sitting next to me than like a search engine.

At the start, I asked it to compare possible projects. It suggested Oxford-IIIT Pet with binary segmentation. I decided to go with three classes (pet, background and boundary) instead, because I wanted the harder version of the task. I also chose to train on Kaggle after it explained the differences between the free Kaggle and Colab GPUs.

After that, we worked in parts: data checks, splits, baseline, training, evaluation, optimization, app, and finally documentation.

The AI also helped me with things that had nothing to do with machine learning but still took a lot of time:
- My project ended up nested inside a second folder, so Git wasn't working where I expected.
- My Mac was logged in to a different GitHub account, so pushes were rejected with a 403 error.
- Later my Mac's disk filled up in the middle of running the tests, and I had to clean up caches and old files.

## How I checked that the AI's output was correct

I didn't want to submit anything I hadn't seen working myself, so I relied on a few things:

- **Tests.** The project has 53 automated tests, and I ran them on my own Mac after every part. They all pass. Some of them check things I wouldn't have thought to test myself. One example: after a random crop and flip, the mask must still line up with the image.
- **Looking at the pictures.** I opened every visual check: the masks drawn on the images, the augmented examples, the duplicate pairs, and the best and worst predictions. This is how I noticed that several of the "worst" test images actually have wrong ground-truth labels (an empty mask, or the background marked as the pet), while the model's prediction looks right.
- **Real numbers only.** Every number in the README and reports comes from a script I ran myself, either locally or on Kaggle. Nothing was typed in by hand. I also checked that the numbers made sense. For example, the pretrained model scored exactly 0 on the boundary class, which is expected because it has no such class, and the validation and test results ranked the models in the same order.

## Where the AI got things wrong

It wasn't perfect, and some of its mistakes only showed up when I actually ran things:

- The `.gitignore` it gave me had the line `data/`, meant to keep the dataset out of GitHub. But it also matched my source folder `src/petseg/data/`, so that code never got pushed. Everything worked on my Mac, but training on Kaggle failed with `ModuleNotFoundError`. We fixed it by changing the pattern to `/data/`.
- It once told me to expect 33 passing tests when the real number was 30. That's small, but it reminded me to trust the actual output over what I was told to expect.
- Some of its instructions assumed I knew more than I did, for example where to find settings in PyCharm or Kaggle. I had to ask follow-up questions several times.

Its experiment idea also didn't work out: the class-weighted loss (E3) was supposed to improve the boundary class, but it made every metric worse. I kept that result in the report instead of hiding it, because a negative result is still a result.
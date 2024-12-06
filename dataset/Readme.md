# Dataset

A custom dataset was created by creating images using StyleGAN, and profile text information from a set of traits generated for each profile (randomly sampeld form a fixed set of values per trait). 

The generated text profiles are processed through GPT-4o to obtain the personality and a profile introduction. We also
add text traits which are implicit to make the personalities more evident and unique.

## Organization

There are a total of 30 profiles for Men and Woman each. 
The structure of the dataset is:

root
├── Man
│   ├── man_profile_1
│   │   ├── cls_token.pt
│   │   ├── profile_1.png
│   │   ├── text_profile_1.json
│   │   └── text_profile_1.txt
|   .
|   .
└── Woman
    ├── woman_profile_1
    │   ├── cls_token.pt
    │   ├── profile_1.png
    │   ├── text_profile_1.json
    │   └── text_profile_1.txt
    .
    .
    .


## Images

The images generated with StyleGAN were scraped from this website - 'https://thispersondoesnotexist.com/'.

## Text

personality_generator.py is used to create personalities for each profile.
# Context
This is a baseline implementation of the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) question and answering system. The goals with this repo is to learn about some of the challenges associated with building NLP systems and also to provide a starting point from which to create a more general Q & A dataset based on unlabeled data.

## Set Up

1. Run the set up shell script to ensure that you have all of the local dependencies needed. `sh set_up.sh`. Note, this script is based on `pip3` and you may need to modify if if you are using `conda` or some other package management system.

1. Run `python3 setup.py` to get all of the data into a usable format.

## Super Context Experiment

Please email me at rmusti@redhat.com if you want to learn more about the idea. I might try and turn this into a research paper so be advised that that may be coming soon.

The idea is to combine all of the contexts into one large "super context".
Then train the question and answering system using the "super context" as the context for every question and answer pair.
I will train on the test set and see how I perform on the dev set.

## Repo Set Up

This repo is divided into exp_1, exp_2, etc. folders and an exp_x folder.
The exp_1, exp_2, etc. folders all contain "memory inefficient" experiments that would not be used in a production setting.
They are primarily divised to see how well this idea of context merging works.
The exp_x folder contains the code to create memory efficient "production" versions of this code.
This generally means that they are more memory efficient and have more robust data ingestion setups.

I originally started on the exp_x work and have shifted to the other experiments for now.
The overhead/headache of the efficiency adjustments outweighed any benefits I might be gaining from it.
I will return to it when I have more information about what the most effective approaches are.

### Experiment 1

This experiment is running the original SQuAD baseline implementation with 1 change: each context has 10 other random contexts tied to it.

### Experiment 2

This experiment is running the original SQuAD baseline implementation with 1 change: each context has all the contexts from its topic merged together into one.

### Experiment 3

This experiment is running the original SQuAD baseline implementation with 1 change: each context has all the contexts merged together.

### Data Re-Structure

#### Original Data Structure (exp_x efficiency adjustments)

The squad data comes in a json format. Run `python3 data_discovery.py` to verify this.


At the highest level, it has two fields `version` and `data`.

- `version`: string indicating whether the data is v1 or v2; we are working with v2.

- `data`: a list of dictionaries and contains all of the data; each dictionary has two keys: `title` and `paragraphs`.

    - `title`: title of the article the paragraphs came from.

    - `paragraphs`: a list of dictionaries with two keys: `qas` and `context`.

        - `context`: a string representing a context paragraph.

        - `qas`: a list of dictionaries. Each dictionary as 4 keys: `id`, `is_impossible`, `question`, `answers`.

            - `id`: just a string representing the id of the question

            - `is_impossible`: a boolean that is True if it answerable and False if not

            - `question`: a string representing a question

            - `answers`: a list of dictionaries, each with two fields: `text`, `answer_start`.

                - `text`: a string indicating the text of a valid answer

                - `answer_start`: an integer indicating the start a valid answer.:w

#### Experiment 2 (topic contexts) Restructure

Restructuring steps:

1. Merge all the `contexts` within each `topic` into one giant string
1. I need to update `answer_start` indexes to account for their movement based on their new context location.
1. I can also be more space efficient by storing the new `topic_context` above the paragraphs on the same level as its corresponding `topic`.
1. Within `paragraphs`, I can make all the elements of `qas` elements of `paragraph` and  rename `paragraphs` to `qas`.

#### Experiment 3 (super contexts) Restructure

Restructuring steps:

1. Merge all the `contexts` into one giant string
1. I need to update `answer_start` indexes to account for their movement based on where they're appended.
1. I can also be more space efficient by storing the new `super_context` above the paragraphs.
1. Within `paragraphs`, I can make all the elements of `qas` elements of `paragraph` and rename `paragraphs` to `qas`.

### Experiment 2 set up

test  what the softmaxed entries of the similarity matrix look like when you train with larger context; so a distribution of the softmaxed entries (row-wise and column-wise) across all questions in (train, dev) for (usual context, larger context sizes)

1. Convert word indices to word embeddings for the context and the questions.
    - efficiency here, I only need to calculate the context embeddings once
1. Project each embedding vector into dimensionality H.
    - to do this, you multiply each vector by a matrix of dimensions H x V, this will be a learnable matrix of parameters.
1. Then, we apply a highway network to transform each hidden vector using learnable parameters.
1. Encoder Layer
1. Attention Layer
1. Modeling Layer


## Credits

- Thank you to @chrischute for his work in creating the `layers.py`, `setup.py` files and setting up the original model I started training from.

- Thank you to Knowledge Computation Group@HKUST for their original code [here](https://github.com/HKUST-KnowComp/R-Net) that I borrowed a lot of structure from.

- Thank you to Sanjay Arora for his mentorship throughout this effort and the Red Hat AI Center of Excellence for making this research project possible.

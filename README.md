dssgsummit2020-challenge
==============================

1st place solution for the DSSG Summit 2020 Challenge: Keyboard layout optimization for ALS patients 

More information about the challenge at https://github.com/nilg-ai/dssgsummit2020-challenge.

The challenge
------------
This project aims at finding an optimal keyboard layout to minimize the workload for usage by ALS patients. More specifically, given a fixed hexagonal keyboard, we optimize for the allocation of keys in order to reduce the total distance traveled between keys to write a large corpus of text using the proposed keyboard layout. The problem translates to finding the best permutation, which is a well-known established optimization problem. 

![alt text](https://raw.githubusercontent.com/nilg-ai/dssgsummit2020-challenge/master/images/image1.png "Keyboard Optimization")

Our solution
------------
Our approach relies on Genetic Algorithms to generate keyboard solutions. This kind of algorithms emulate the biological process of natural selection, where the principle of survival of the fittest applies. We experimented with several hyperparameter configurations, different mutation, crossover and selection algorithms. 

The best solutions we found both for the Portuguese and English keyboard can be seen below:

**INSERIR KEYBOARD DAS NOSSAS MELHORES SOLUÇÕES!**

Reproducing results
------------
### Data
A large and diverse corpus of text is necessary for evaluating each solution we obtain as we want a general purpose keyboard that is able to retain its good performance on a multitude of writing tasks, from simple communication to more structured compositions.

Download the english corpus by using the following command: `curl http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip --output data/raw/bbc-fulltext.zip; unzip data/raw/bbc-fulltext.zip`. To download the portuguese corpus you first need to obtain a username and password by registering on https://www.linguateca.pt/cetempublico/informacoes.html#registo. Once you receive an email with the username and password, use `curl -u <username> https://www.linguateca.pt/cetempublico/download/CETEMPublico1.7.gz --output data/raw/CETEMPublico1.7.gz; gunzip data/raw/CETEMPublico1.7.gz`, replace \<username\> with the actual username sent in the email and then pass the password when prompted.

Keep in mind that the data has a combined size of ~1.2G (mainly because of the portuguese corpus).
### Environment
The necessary project dependencies can be installed through a python environment and the requirements.txt file. You can create the environment using `make environment`. Activate the environment and then `make requirements` to install all necessary packages. 

### Code
The first step is to preprocess our data so the cost function can be computed and give reasonable results. This can be achieved by running `make data`. 

To start running experiments using Genetic Algorithms and obtain new solutions based on different hyperparameter settings you can run the `ga_keyboard_experiment.py` script under dssg_challenge/ga/ and you can also specify which hyperparameters to explore in a grid placed in this file.

To visualize the keyboard given by one selected solution you can run the `draw_keyboard.py` under dssg_challenge/ga/.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    |   └── raw            <- The original, immutable data dump.
    |
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    |
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    |
    ├── dssg_challenge          <- Source code for use in this project.
    │   ├── __init__.py         <- Makes dssg_challenge a Python module
    │   ├── _concat_logs.py
    │   ├── compute_cost.py
    │   ├── corpus_preprocessing.py
    │   ├── draw_keyboard.py
    │   ├── ga_keyboard_experiment.py
    │   ├── ga                  <- Genetic Algorithm implementation and building blocks
    │   │   ├── algorithm
    │   │   ├── custom_problem
    │   │   ├── problem
    │   │   ├── test
    │   │   ├── toy_problem
    │   │   └── util
    │   └── utils.py
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    |
    ├── setup.py           <- Makes project pip installable (pip install -e .) so dssg_challenge can be imported
    |
    └── test_environment.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

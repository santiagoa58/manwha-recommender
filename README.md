# Manwha Recommender System

## Overview

The Manwha Recommender System is a command-line tool that recommends Manwha based on a given title. It uses a K-Nearest Neighbors model trained on TF-IDF vectors of Manwha descriptions and tags to find similar Manwhas.

## Table of Contents

- [Installation](#installation)
- [Building the Model](#building-the-model)
- [Using the Recommender](#using-the-recommender)
- [Project Structure](#project-structure)

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/santiagoa58/manwha-recommender.git
   cd manwha-recommender
   ```

2. **Set Up a Virtual Environment** (Optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Building the Model

Before using the recommender, you need to build the model.

```bash
python -m src.utils.build_recommender
```

This will preprocess the data, train the model, and serialize it for later use.

## Using the Recommender

Once you've built the model, you can get recommendations for a Manwha using the CLI:

```bash
python -m src.cli.main
```

Follow the prompts and enter the title of the Manwha you're interested in.

## Project Structure

- `data/`: Folder containing the raw and preprocessed data.
- `models/`: Folder containing the serialized model.
- `src/`: Main source code directory.
  - `cli/`: Command line scripts to interface with the recommender.
  - `recommender/`: Core recommender system code.
  - `utils/`: Utilities for building the recommender and other auxiliary functions.
- `requirements.txt`: Project dependencies.

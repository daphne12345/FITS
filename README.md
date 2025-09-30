# FITS: Towards an AI-Driven Fashion Information Tool for Sustainability

This repository contains the code for the research paper titled "FITS: Towards an AI-Driven Fashion Information Tool for Sustainability" accepted at *ECAI 2025*. The code implements the methods and experiments described in the paper.

## Overview
Access to credible sustainability information in the fashion industry remains limited and challenging to interpret, despite growing public and regulatory demands for transparency. General-purpose language models often lack domain-specific knowledge and tend to "hallucinate", which is particularly harmful for fields where factual correctness is crucial. This work explores how Natural Language Processing (NLP) techniques can be applied to classify sustainability data for fashion brands, thereby addressing the scarcity of credible and accessible information in this domain. We present a prototype Fashion Information Tool for Sustainability (FITS), a transformer-based system that extracts and classifies sustainability information from credible, unstructured text sources: NGO reports and scientific publications. Several BERT-based language models, including models pretrained on scientific and climate-specific data, are fine-tuned on our curated corpus using a domain-specific classification schema, with hyperparameters optimized via Bayesian optimization. FITS allows users to search for relevant data, analyze their own data, and explore the information via an interactive interface. We evaluated FITS in two focus groups of potential users concerning usability, visual design, content clarity, possible use cases, and desired features. Our results highlight the value of domain-adapted NLP in promoting informed decision-making and emphasize the broader potential of AI applications in addressing climate-related challenges. Finally, this work provides a valuable dataset, the SustainableTextileCorpus, along with a methodology for future updates.  


## Installation

You can install the necessary packages using the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

# Usage

## Creation of the SustainableTextileCorpus
Theoretically our method works on any PDF text files that contain keywords in the [keyword list](data/kw_list.csv) and a brands from the [brand list](data/brand_list.csv). 
To recreate the SustainableTextileCorpus, please use the provided literature list and download the respectice articles or use a keyword search, as descibed in the paper, to get more recent relevant articles.

Each of the scripts conatin a main method, so they can all be run consecutively:
```
cd 3_Creation_SustainableTextileCorpus
```

1. To check for a missing DOI and license, adjust the path to the file containing the abstracts and run this: 
    ```
    python 3_2_scientifc_publications.py
    ```

2. Download the NGO reports: 
    ```
    python 3_2_ngo_reports.py
    ```

3. To preprocess the pdfs and abstracts, run: 
    ```
    python 3_3_preprocessing.py
    ```

4. To label some of the data, run this and label the data in LightTag (this also contains the code to read the labeling from Lighttag): 
    ```
    python 3_4_Labeling.py
    ```

5. To analyse the data and recreate the plots in the paper, run: 
    ```
    python 3_5_Analysis.py
    ```

## Model Training and Optimization
This part recreates the result in the paper.
```
cd 4_Model_Training_and_Optimization
```

1. To train the HPO baseline model (BERT for 20 epochs with the default hyperparameters), run: 
    ```
    python 4_2_Model_Training.py
    ```

2. To train the baselines, run: 
    ```
    python 4_3_Baselines/baseline_kw_matching.py
    ```
    and:
    ```
    python 4_3_Baselines/baseline_tfidf_svm.py
    ```

3. To perfrom hyperparameter optimization, run: 
    ```
    python 4_4_HPO.py --model_name roberta-base --seed 0
    ```

## Run FITS
In addition to some relevant PDF files or the SustainableTextileCorpus, a path to a checkpoint of a fine-tuned model needs to be inserted in the [preprocessing file](5_Tool/preprocessing.py).


To start the application, run:

 ```python 5_Tool/app.py```

This starts a flask application that could, for example, run on a server and several users can access it at the same time.

# Citation 
```
@article{fits,
  title={FITS: Towards an AI-Driven Fashion Information Tool for Sustainability},
  author={Daphne Theodorakopoulos, Elisabeth Eberling, Miriam Bodenheimer, Sabine Loos and Frederic Stahl},
  booktitle={ECAI 2025},
  year={2025},
  publisher={IOS Press},
  note={in press}
} 
```

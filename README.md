# LLM-Abstractive-Summarization

## Project Description
This project focuses on performing abstractive summarization using a pre-trained Large Language Model (LLM). Specifically, it involves fine-tuning the `facebook/bart-base` model on two distinct datasets: Billsum and Multi-News, and evaluating its performance using standard summarization metrics.

## Features
* Fine-tuning of the `facebook/bart-base` LLM for abstractive summarization.
* Data preprocessing for the Billsum and Multi-News datasets.
* Evaluation using ROUGE, BLEU, and BERTScore metrics.
* Demonstrates the application of LLMs for generating concise summaries from longer texts.

## Datasets Used
The project utilizes the following datasets for training and evaluation:
* **Billsum**: Contains summaries of US Congressional and California state bills.
    * Number of training samples: 18949
    * Number of test samples: 3269
    * Average document length (words): 1289.39
    * Average summary length (words): 179.12
    * Vocabulary size before tokenization: 361685
* **Multi-News**: Consists of news articles and human-written summaries of these articles.

## Model Used
The core model for this task is `facebook/bart-base` from Hugging Face Transformers. The fine-tuning process considers mixed precision training using `bfloat16` and adjusts batch size to accommodate the model into the GPU.

## Evaluation Metrics
The performance of the fine-tuned LLM is evaluated using the following widely accepted summarization metrics:
* **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation)
* **BLEU** (Bilingual Evaluation Understudy)
* **BERTScore**

## Expected Scores (Test Sets)
The project aims to achieve the following performance benchmarks:
* **Billsum:**
    * Rouge-1: >40
    * Rouge-2: >18
    * Rouge-L: >28
    * BLEU: >12
    * BERTScore: >75
* **Multinews:**
    * Rouge-1: >35
    * Rouge-2: >5
    * Rouge-L: >13
    * BLEU: >3.5
    * BERTScore: >75

## Setup and Installation
To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/LLM-Abstractive-Summarization.git](https://github.com/your-username/LLM-Abstractive-Summarization.git)
    cd LLM-Abstractive-Summarization
    ```
2.  **Install dependencies:**
    The project requires the following Python libraries. You can install them using pip:
    ```bash
    pip install transformers datasets rouge-score sacrebleu bert-score accelerate evaluate
    ```
    *(Note: Ensure you have a suitable environment with GPU support for optimal performance with `bf16` precision.)*

## Usage
Detailed instructions on how to run the fine-tuning process, perform summarization, and evaluate the models will be provided in the Jupyter Notebooks within this repository.

## Results
During preliminary validation, the model achieved the following scores on a subset of the test data after fine-tuning for 5 epochs:
* ROUGE Scores:
    * rouge1: 0.4267
    * rouge2: 0.2557
    * rougeL: 0.3288
    * rougeLsum: 0.3621
* sacreBLEU score: 10.3442
* BERTScore:
    * Precision: 0.3869
    * Recall: 0.2044
    * F1: 0.2919

# LLM-Abstractive-Summarization

## Project Description
This project focuses on performing abstractive summarization using a pre-trained Large Language Model (LLM). It involves fine-tuning the `facebook/bart-base` model on two distinct datasets: Billsum and Multi-News, and evaluating its performance using standard summarization metrics such as ROUGE, BLEU, and BERTScore. The project explores the model's capabilities across different domains of text summarization.

## Features
* Fine-tuning of the `facebook/bart-base` LLM for abstractive summarization.
* Data preprocessing for the Billsum and Multi-News datasets.
* Evaluation using ROUGE, BLEU, and BERTScore metrics.
* Demonstrates the application of LLMs for generating concise summaries from longer texts across different text genres (bills and news articles).

## Datasets Used
The project utilizes the following datasets for training and evaluation:
* **Billsum**: Contains summaries of US Congressional and California state bills.
    * Number of training samples: 18949
    * Number of test samples: 3269
    * Average document length (words): 1289.39
    * Average summary length (words): 179.12
    * Vocabulary size before tokenization: 361685
* **Multi-News**: Consists of news articles and human-written summaries of these articles. This dataset was specifically used for the bonus summarization task.

## Model Used
The core model for this task is `facebook/bart-base` from Hugging Face Transformers. The fine-tuning process considers mixed precision training using `bfloat16` and adjusts batch size to accommodate the model into the GPU.

## Evaluation Metrics
The performance of the fine-tuned LLM is evaluated using the following widely accepted summarization metrics:
* **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation)
* **BLEU** (Bilingual Evaluation Understudy)
* **BERTScore**

## Expected Scores (Test Sets)
The project aims to achieve the following performance benchmarks for the Billsum dataset:
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

## Bonus Task: Multi-News Summarization
This section details an additional task where the summarization process from Part IV was repeated using the Multi-News dataset. This involved dataset preparation, preprocessing, and fine-tuning the BART model to generate summaries for news articles. The resulting model and tokenizer were saved and subsequently hosted on Hugging Face.

### Hosted Model
The fine-tuned BART model for the Multi-News dataset is publicly available on Hugging Face:
[**Pavithran27/bart-multinews**](https://huggingface.co/Pavithran27/bart-multinews)

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
Detailed instructions on how to run the fine-tuning process, perform summarization, and evaluate the models will be provided in the Jupyter Notebooks within this repository. The main summarization task is covered in `a2_part_4_pgnanase_shebbar (1).ipynb`, and the bonus task using Multi-News is in `a2_bonus_summary_pgnanase_shebbar_FINAL_last.ipynb`.

You can also directly load and use the fine-tuned Multi-News model from Hugging Face:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "Pavithran27/bart-multinews"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Example usage (assuming you have an input text)
# input_text = "Your long news article here..."
# inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
# summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True)
# summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# print(summary)

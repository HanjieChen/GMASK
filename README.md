# GMASK
Code for the paper "Explaining Neural Network Predictions on Sentence Pairs via Learning Word-Group Masks"

### Requirement:
- python == 3.6.11
- pytorch == 1.4.0
- numpy == 1.18.5

### Data:
Download the [data](https://drive.google.com/drive/folders/1J18AsUKuBYFtHmV0b1pfyd93G_lb2eLQ?usp=sharing) and put it in the same folder with the code.

### Train VMASK Models:

For BERT-VMASK, we adopt the BERT-base model built by huggingface: https://github.com/huggingface/transformers. We first trained BERT-base model, and then loaded its word embeddings for training BERT-VMASK. You can download our [pretrained BERT-base models](https://drive.google.com/drive/folders/19nzAv1wWtM5UCNAORMqrNtBH6j3ZXmSz?usp=sharing), and put them under the same folder with the code.

In each folder, run the following command to train VMASK-based models.
```
python main.py --save /path/to/your/model
```
Fine-tune hyperparameters (e.g. learning rate, the number of hidden units) on each dataset.

### Reference:
If you find this repository helpful, please cite our paper:
```bibtex
@inproceedings{chen-etal-2021-explaining,
    title = "Explaining Neural Network Predictions on Sentence Pairs via Learning Word-Group Masks",
    author = "Chen, Hanjie  and
      Feng, Song  and
      Ganhotra, Jatin  and
      Wan, Hui  and
      Gunasekara, Chulaka  and
      Joshi, Sachindra  and
      Ji, Yangfeng",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.306",
    pages = "3917--3930",
    abstract = "Explaining neural network models is important for increasing their trustworthiness in real-world applications. Most existing methods generate post-hoc explanations for neural network models by identifying individual feature attributions or detecting interactions between adjacent features. However, for models with text pairs as inputs (e.g., paraphrase identification), existing methods are not sufficient to capture feature interactions between two texts and their simple extension of computing all word-pair interactions between two texts is computationally inefficient. In this work, we propose the Group Mask (GMASK) method to implicitly detect word correlations by grouping correlated words from the input text pair together and measure their contribution to the corresponding NLP tasks as a whole. The proposed method is evaluated with two different model architectures (decomposable attention model and BERT) across four datasets, including natural language inference and paraphrase identification tasks. Experiments show the effectiveness of GMASK in providing faithful explanations to these models.",
}
```

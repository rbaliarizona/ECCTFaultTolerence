# Fault Tolerance Testing of Transformer-Based Error Correction Models

## Overview  
This project examines the fault tolerance of transformer-based error correction models, focusing on noise introduced during and due to syndrome processing. It builds on the transformer architecture proposed in [*Accelerating Error Correction Code Transformers*](https://arxiv.org/pdf/2410.05911v1).

This is part of the final project for  **ECE 537: Coding and Information Theory**, Fall 2024, taught by Dr. Bane Vasic and Dr. Asit Kumar Pradhan [Course Link](https://ece.engineering.arizona.edu/course/ece/537).  

## Key Resources  
- **Overleaf Document**: Contains the project writeup and current analysis. [Access here](https://www.overleaf.com/project/6744fbbdfa274c289f00070c).  

## Setup

1. **Clone the repository**:  
    ```bash
    git clone https://github.com/yourusername/project-repo.git
    cd project-repo
    ```

2. **Install dependencies**:  
    If you haven’t already, install the required libraries:  
    ```bash
    pip install -r requirements.txt
    ```

### Training a decoder  

Use the following command to train a 6 layers AECCT of dimension 128 on the LDPC(49,24) code:

```bash
python main.py --code LDPC_N49_K24 --N_dec 6 --d_model 128
```




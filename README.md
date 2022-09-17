# IoT-Activity-Detection-Clean
Updating

## Transformer/model.py
This file is for building the Transformer model. Three functions `transformer_base`,
`transformer_large` and `transformer_huge` return three Transformer models with
different size(different number of parameters).  
"""python
model_base = transformer_base()
model_large = transformer_large()
model_huge = transformer_huge()
"""
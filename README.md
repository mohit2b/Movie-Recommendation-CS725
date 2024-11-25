Under main model -

- for gcn_model.py
  - change the other model parameters like learning rate, hidden_dimension, output_dimension in this file as you like
  - provide the following details
  - ```
	PATH = "path to save model"
	loss_name = "bpr" # options: bpr, margin_loss
	model_type = "GCN" # options: GAT, GCN
    ```
- for gcn_model_random_lsh.py--
  - provide the following details
  -   ```
  PATH = "path to load the model"
  model_type = "GCN"
    ```

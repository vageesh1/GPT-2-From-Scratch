# GPT-2-from-sctach
Making GPT-2 from scracth with modifications of ROPE, sliding window attention and grouped query attention<br>
## Features

- **Rotational Positional Embedding:** Introduces rotational positional embeddings to improve the model's understanding of token positions.

- **Sliding Window Attention:** Implements sliding window attention to focus on local contexts, allowing the model to capture more fine-grained information.

- **Multi-Query Attention:** Enhances attention mechanisms with multi-query attention, enabling the model to consider multiple queries simultaneously.

## Training Results

After extensive experimentation, it was observed that a context size of 25 yielded promising results compared to the original architecture. The following steps outline the training process:

1. **Data Preparation:**
   - For this project, the training data utilized Shakespearean text for tokenization. Ensure your dataset is preprocessed and tokenized using the provided Shakespearean text embeddings or any other suitable embeddings.

2. **Model Configuration:**
   - Adjust the configuration file to include your modifications (rotational positional embedding, sliding window attention, multi-query attention). Additionally, introduce random model configurations to add an element of variability.

3. **Training:**
   - Train the model using the following command:
     ```bash
     python train.py --context_size 25 --embedding_type shakesperean --random_config --other_parameters...
     ```

4. **Evaluation:**
   - Evaluate the model on your test set to assess its performance.

## Experimentations

After extensive experimentation, the custom GPT-2 architecture demonstrated notable advantages over the original architecture, especially when utilizing a smaller context window. Here are some key findings:

1. **Context Window Optimization:**
   - The introduction of sliding window attention, rotational positional embedding, and multi-query attention allowed for effective training with a smaller context window (context size 25). Despite the reduced context, the model exhibited enhanced understanding and capturing of intricate dependencies within the input data.

2. **Token Attention Efficiency:**
   - The customized attention mechanisms ensured that all tokens within the specified context received attention, leading to a more comprehensive understanding of local contexts. This improvement contributed significantly to the model's ability to capture nuanced relationships and generate more contextually relevant outputs.

3. **Reduced Inference Time:**
   - The optimized architecture not only excelled in training but also showcased reduced inference time during text generation. The combination of sliding window attention and multi-query attention led to more efficient computations, making the model well-suited for real-time or resource-constrained applications.

4. **Performance Outcomes:**
   - Comparative evaluations against the original GPT-2 architecture consistently demonstrated superior performance in terms of training convergence, overall model quality, and efficiency in utilizing smaller context windows, a slightly bigger size than original model

## Example Usage

Here's a simple example of how to use the trained model for text generation:

```python
#to load the final modified architecture
from gpt2_rope_slide_multiquery import Attention_rope_slide_group
from training_loop import TrainerConfig,Trainer,CharDataset,top_k_logits,sample


# make an instance of model
model = Attention_rope_slide_group()

#loading dataset
block_size=25

text = open('/content/drive/MyDrive/shakespeare.txt', 'r').read()
train_dataset = CharDataset(text, block_size = 25) # 25 is for context storing

#for training on single GPU, multi GPU needs to be tested later
model=model.to(device)
tconf = TrainerConfig(
    max_epochs=1,
    batch_size=8,
    learning_rate=6e-4,
    lr_decay=True,
    warmup_tokens=512,
    final_tokens=2*len(train_dataset)*block_size,
    num_workers=4,
)
trainer = Trainer(model, train_dataset, None, tconf)
trainer.train()


# Generate text
context = "When are you going to change the diaper?"

# Check if each character in the context is in the vocabulary
try:
    indices = [train_dataset.stoi[s] for s in context]
except KeyError as e:
    print(f"Error: Character '{str(e)}' not found in the vocabulary.")
    # Handle the error or skip this context

# Convert the indices to a torch tensor
x = torch.tensor(indices, dtype=torch.long)[None, ...].to(trainer.device)

# Rest of your code...
y = sample(trainable_model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
completion = ''.join([train_dataset.itos[int(i)] for i in y])
print(completion)

#!/usr/bin/env python
# coding: utf-8

# # DLCV (DS-265) Assignment 2
# # Vision Transformer using Pytorch
# 
# ## Name - Rishav Saha
# ## Sr No - 22573

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision import transforms
import math
from torch.utils.data import random_split
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import torch.nn as nn

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        # Check if embed_dim can be divided evenly into num_heads
        assert embed_dim % num_heads == 0, f"Can't divide dimension {embed_dim} into {num_heads} heads"
        
        # Initialize parameters
        self.embed_dim = embed_dim  # Embedding dimension of input sequences
        self.num_heads = num_heads  # Number of attention heads
        self.d_head = embed_dim // num_heads  # Dimension of each head
        
        # Multi-head attention mechanism
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, sequences):
        # Input sequences shape: (seq_length, N, embed_dim)
        sequences = sequences.permute(1, 0, 2)  # Permute to (N, seq_length, embed_dim)
        
        # Apply multi-head attention mechanism
        attn_output, attn_weights = self.multihead_attn(sequences, sequences, sequences)
        
        # Permute output back to original shape: (seq_length, N, embed_dim)
        attn_output = attn_output.permute(1, 0, 2)
        
        return attn_output, attn_weights


# In[3]:


import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, num_heads, hidden_dimension, mlp_ratio=4):
        super().__init__()
        
        # Initialize parameters
        self.num_heads = num_heads
        self.hidden_dimension = hidden_dimension
        
        # Layer normalization before attention
        self.normalize_layer_1 = nn.LayerNorm(hidden_dimension)
        
        # Multi-headed self-attention mechanism
        self.multi_headed_self_attention = MultiHeadedSelfAttention(num_heads=num_heads, embed_dim=hidden_dimension)
        
        # Layer normalization after attention
        self.normalize_layer_2 = nn.LayerNorm(hidden_dimension)
        
        # Feed-forward neural network
        self.neural_network = nn.Sequential(
            nn.Linear(hidden_dimension, mlp_ratio * hidden_dimension),  # Linear transformation
            nn.GELU(),  # Activation function
            nn.Linear(mlp_ratio * hidden_dimension, hidden_dimension)  # Linear transformation
        )

    def forward(self, encoder_input):
        # Layer normalization before attention
        norm_layer1_output = self.normalize_layer_1(encoder_input)
        
        # Multi-headed self-attention mechanism
        self_attention_output, attention_weights = self.multi_headed_self_attention(norm_layer1_output)
        
        # Layer normalization after attention
        norm_layer2_output = self.normalize_layer_2(self_attention_output)
        
        # Feed-forward neural network
        neural_network_output = self.neural_network(norm_layer2_output)
        
        # Final output of the encoder block
        final_output = neural_network_output
        
        return final_output, attention_weights


# In[4]:


import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, patch_size, image_size, hidden_dimension, num_heads, out_dimension, num_encoder_blocks=2):
        super().__init__()
        # Initialize parameters
        self.patch_size = patch_size
        self.hidden_dimension = hidden_dimension
        self.num_heads = num_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.output_dimension = out_dimension
        self.input_dimension = int(3 * self.patch_size * self.patch_size)  # Dimension of input patches after flattening
        
        # Linear embedding layer
        self.linear_mapper = nn.Linear(self.input_dimension, self.hidden_dimension)
        
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, self.hidden_dimension))
        
        # Compute number of patches
        batch_size, channel, height, width = image_size
        num_of_patches = (height * width) // (self.patch_size ** 2)
        
        # Generate positional encodings
        self.positional_encodings = self._generate_positional_encodings(int(num_of_patches), self.hidden_dimension)
        
        # Create a list of encoder blocks
        self.blocks = nn.ModuleList([EncoderBlock(hidden_dimension=self.hidden_dimension, num_heads=self.num_heads) for _ in range(self.num_encoder_blocks)])
        
        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dimension, self.output_dimension),  # Linear layer
            nn.Softmax(dim=-1)  # Softmax activation
        )
    
    def _generate_positional_encodings(self, num_patches, hidden_dimension):
        # Create positions from 0 to num_patches - 1
        positions = torch.arange(0, num_patches).unsqueeze(1)
        
        # Compute the angles for even and odd indices
        angles = torch.arange(0, hidden_dimension, 2).float() / hidden_dimension
        angles = 1 / torch.pow(10000, angles)
        
        # Compute the positional encodings
        positional_encodings = torch.zeros(num_patches, hidden_dimension)
        positional_encodings[:, 0::2] = torch.sin(positions.float() * angles)
        positional_encodings[:, 1::2] = torch.cos(positions.float() * angles)
        
        # Add the class token positional encoding
        positional_encodings = torch.cat([torch.zeros(1, hidden_dimension), positional_encodings])
        
        return positional_encodings.unsqueeze(0)
    
    def forward(self, input_images):
        # Split the input images into patches
        batch_size = input_images.shape[0]
        image_patches = input_images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        
        # Reshape the patches
        image_patches = image_patches.contiguous().view(input_images.size(0), -1, self.patch_size * self.patch_size * input_images.size(1))
        tokens = self.linear_mapper(image_patches)
        
        # Add the class token to the tokens
        tokens_with_class = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        
        # Add positional encodings
        tokens_with_position = tokens_with_class + self.positional_encodings[:, :tokens_with_class.size(1)]
        
        attention_weights = []
        # Pass through encoder blocks
        for block in self.blocks:
            tokens_with_position, attn_weight = block(tokens_with_position)
            attention_weights.append(attn_weight)
        
        # Get only the classification token
        tokens_with_position = tokens_with_position[:, 0]
        
        # Apply the MLP classifier
        return self.mlp(tokens_with_position), attention_weights


# In[5]:


BATCH_SIZE=64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(
    'train_data',
    transform=transform,
    target_transform=transforms.Compose([
        lambda x: torch.tensor(x),  # Convert label to tensor
    ]),
    train=True,
    download=True
)

test_data = datasets.CIFAR10(
    'test_data',
    transform=transform,
    target_transform=transforms.Compose([
        lambda x: torch.tensor(x),  # Convert label to tensor
    ]),
    train=False,
    download=True
)

test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)


# In[6]:


# Defining model and training options
device = torch.cuda.set_device(1)
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
transformer_model = VisionTransformer(image_size=(BATCH_SIZE,3, 32, 32), num_encoder_blocks=2, hidden_dimension=8, num_heads=2, out_dimension=10,patch_size=3)

# model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
N_EPOCHS = 5
LR = 0.005
criterion = CrossEntropyLoss()
# Test loop
def test(model):
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat,attn = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)
    
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {correct / total * 100:.2f}%")


# In[7]:


import matplotlib.pyplot as plt

N_EPOCHS = 10
LR = 0.005

def train(train_data_loader, model):
    train_loader = train_data_loader
    optimizer = Adam(model.parameters(), lr=LR)
    train_losses = []  # List to store training losses for each epoch
    
    for epoch in range(N_EPOCHS):
        train_loss = 0.0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat, _ = model(x)
            loss = criterion(y_hat, y)
    
            train_loss += loss.detach().cpu().item() / len(train_loader)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_losses.append(train_loss)  # Append the training loss for this epoch
        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Plotting the training loss
    plt.plot(range(1, N_EPOCHS + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

# train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
# train_2(train_loader,transformer_model)


# # Training the model with different data sizes: 5% , 10% , 25% , 50% , 100%

# ## Trainig with 5% training data

# In[108]:


five_percent = int(0.05 * len(train_data))

subset_dataset, _ = random_split(train_data, [five_percent, len(train_data) - five_percent])
model_1=VisionTransformer(image_size=(BATCH_SIZE,3, 32, 32), num_encoder_blocks=2, hidden_dimension=8, num_heads=4, out_dimension=10,patch_size=3)
# Create a DataLoader for the subset dataset
subset_loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=True)
train(subset_loader,model=model_1)


# In[110]:


test(model=model_1)


# ## Training with 10% training data

# In[111]:


ten_percent = int(0.1 * len(train_data))

subset_dataset, _ = random_split(train_data, [ten_percent, len(train_data) - ten_percent])
model_2=VisionTransformer(image_size=(BATCH_SIZE,3, 32, 32), num_encoder_blocks=2, hidden_dimension=8, num_heads=4, out_dimension=10,patch_size=3)
# Create a DataLoader for the subset dataset
subset_loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=True)
train(subset_loader,model=model_2)


# In[112]:


test(model=model_2)


# ## Training with 25% training data

# In[113]:


twenty_five_percent = int(0.25 * len(train_data))

subset_dataset, _ = random_split(train_data, [twenty_five_percent, len(train_data) - twenty_five_percent])
model_3=VisionTransformer(image_size=(BATCH_SIZE,3, 32, 32), num_encoder_blocks=2, hidden_dimension=8, num_heads=4, out_dimension=10,patch_size=3)
# Create a DataLoader for the subset dataset
subset_loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=True)
train(subset_loader,model=model_3)


# In[114]:


test(model=model_3)


# ## Training with 50% training data

# In[115]:


fifty_percent = int(0.5 * len(train_data))

subset_dataset, _ = random_split(train_data, [fifty_percent, len(train_data) - fifty_percent])
model_4=VisionTransformer(image_size=(BATCH_SIZE,3, 32, 32), num_encoder_blocks=2, hidden_dimension=8, num_heads=4, out_dimension=10,patch_size=3)
# Create a DataLoader for the subset dataset
subset_loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=True)
train(subset_loader,model=model_4)


# In[116]:


test(model=model_4)


# ## Training with 100% training data

# In[117]:


subset_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
model_5=VisionTransformer(image_size=(BATCH_SIZE,3, 32, 32), num_encoder_blocks=2, hidden_dimension=8, num_heads=4, out_dimension=10,patch_size=3)
train(subset_loader,model=model_5)


# In[118]:


test(model=model_5)


# # Training with different patch sizes: (4x4) , (8x8) , (16x16)

# ## Training with patch size (4x4)

# In[119]:


model_patch_4=VisionTransformer(image_size=(BATCH_SIZE,3, 32, 32), num_encoder_blocks=2, hidden_dimension=8, num_heads=4, out_dimension=10,patch_size=4)
train_data_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
train(train_data_loader,model=model_patch_4)


# In[120]:


test(model=model_patch_4)


# ## Training with patch size (8x8)

# In[121]:


model_patch_8=VisionTransformer(image_size=(BATCH_SIZE,3, 32, 32), num_encoder_blocks=2, hidden_dimension=8, num_heads=4, out_dimension=10,patch_size=8)
train(train_data_loader,model=model_patch_8)


# In[122]:


test(model=model_patch_8)


# ## Training with patch size (16x16)

# In[123]:


model_patch_16=VisionTransformer(image_size=(BATCH_SIZE,3, 32, 32), num_encoder_blocks=2, hidden_dimension=8, num_heads=4, out_dimension=10,patch_size=16)
train(train_data_loader,model=model_patch_16)


# In[124]:


test(model=model_patch_16)


# # Training with different number of attention heads

# ## Training with 4 attention heads

# In[11]:


model_attention_4=VisionTransformer(image_size=(BATCH_SIZE,3, 32, 32), num_encoder_blocks=2, hidden_dimension=8, num_heads=4, out_dimension=10,patch_size=2)
train_data_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
train(train_data_loader,model=model_attention_4)


# In[12]:


test(model=model_attention_4)


# ## Training with 8 attention heads

# In[13]:


model_attention_8=VisionTransformer(image_size=(BATCH_SIZE,3, 32, 32), num_encoder_blocks=2, hidden_dimension=8, num_heads=8, out_dimension=10,patch_size=2)
train(train_data_loader,model=model_attention_8)


# In[14]:


test(model=model_attention_8)


# ## Training with 12 attention heads

# In[8]:


model_attention_12=VisionTransformer(image_size=(BATCH_SIZE,3, 32, 32), num_encoder_blocks=5, hidden_dimension=12, num_heads=12, out_dimension=10,patch_size=2)
train_data_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
train(train_data_loader,model=model_attention_12)


# In[9]:


test(model=model_attention_12)


# ##  Taking 2 test images per class, classifying them, and visualize attention maps across the trained transformer layers.

# In[68]:


classes = test_data.classes
test_images = {}
for class_name in classes:
    test_images[class_name] = []
for i, (image, label) in enumerate(test_data):
    if len(test_images[classes[label]]) < 2:
        test_images[classes[label]].append(image)
    if all(len(images) == 2 for images in test_images.values()):
        break
model=transformer_model
model.eval()

# Classify test images and visualize attention maps
for class_name, images in test_images.items():
    for idx, image in enumerate(images):
        with torch.no_grad():
            output, attention_weights = model(image.unsqueeze(0))

        fig, axs = plt.subplots(1, len(attention_weights))
        for layer, attn_map in enumerate(attention_weights):
            # Normalize attention weights for visualization
            normalized_attn_map = attn_map.squeeze().cpu().numpy()
            normalized_attn_map = (normalized_attn_map - np.min(normalized_attn_map)) / (np.max(normalized_attn_map) - np.min(normalized_attn_map))
            # Plot attention map
            axs[layer].imshow(normalized_attn_map, cmap='hot', interpolation='nearest')
            axs[layer].set_title(f'Layer {layer+1} Attention Map')
            axs[layer].axis('off')
        plt.suptitle(f'Class: {class_name}, Image: {idx+1}')
        plt.show()


# In[ ]:





# In[ ]:





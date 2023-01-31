###############################################
#
# gan128.py
#
# Generative adversarial network (GAN) tutorial
#
# Luke Sheneman
# sheneman@uidaho.edu
# January 2023
#
###############################################


import os

import torch
import torch.nn as nn
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import torchvision.transforms as tt

from torchsummary import summary

from tqdm import tqdm




#
# Define model and training hyperparameters
#
EPOCHS=10000
IMAGE_SIZE=128
BATCH_SIZE=256
STATS = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
LATENT_SIZE=128
LEARNING_RATE=0.0002
NUM_WORKERS=20


#
# Define our input and output directories
#
INPUT_DIR  = "faces"
OUTPUT_DIR = "output128"

#
# Define where to save/load our trained models
#
GENERATOR_MODEL     = "./generator_gan128.pt"
DISCRIMINATOR_MODEL = "./discriminator_gan128.pt"


#
# Return a GPU torch device if we detect a GPU
#
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')



#
# Move tensor(s) to GPU
#
def to_device(data, device):
	if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]

	return data.to(device, non_blocking=True)



#
# Convert tensor(s) back to usable image
#
def denormalize(img_tensors):
  return img_tensors * STATS[1][0] + STATS[0][0]



#
# Wrap a dataloader for the chosen device (GPU or CPU)
#
class DeviceDataLoader():
	def __init__(self, dl, device):
		self.dl = dl
		self.device = device

	# yield a batch of data after moving it to device
	def __iter__(self): 
		for b in self.dl:
			yield to_device(b, self.device)

	# number of batches
	def __len__(self):
		return len(self.dl)


#
# Determine which device to use (CPU vs GPU)
#
device = get_default_device()
print("CHOSEN DEVICE: ", device)


#
# DISCRIMINATOR
#
# Define fully convolutional Discriminator network architecture
# This is a 6-layer CNN expecting a 3x128x128 input tensor and outputting a single scalar
# Use stride=2 to do pooling
#
discriminator = nn.Sequential(

	# input layer:  tensor is 3x128x128 image (as normalized tensor)
	nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1,bias=True),
	nn.BatchNorm2d(64),
	nn.LeakyReLU(0.2, inplace=True),

	# Next Conv Layer:  tensor is 64x64x64 image (as normalized tensor)
	nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=True),
	nn.BatchNorm2d(128),
	nn.LeakyReLU(0.2, inplace=True),

	# Next Conv Layer:  tensor is 128x32x32
	nn.Conv2d(128,256,kernel_size=4, stride=2, padding=1, bias=True),
	nn.BatchNorm2d(256),
	nn.LeakyReLU(0.2, inplace=True),

	# Next Conv Layer:  tensor is 256x16x16
	nn.Conv2d(256,512,kernel_size=4, stride=2, padding=1, bias=True),
	nn.BatchNorm2d(512),
	nn.LeakyReLU(0.2, inplace=True),

	# Next Conv Layer:  tensor is 512x8x8
	nn.Conv2d(512,1024,kernel_size=4, stride=2, padding=1, bias=True),
	nn.BatchNorm2d(1024),
	nn.LeakyReLU(0.2, inplace=True),

	# Last Conv Layer:  tensor is 1024x4x4
	nn.Conv2d(1024,1,kernel_size=4, stride=1, padding=0, bias=True),
	nn.Flatten(),
	nn.Sigmoid() )



#
# GENERATOR
#
# Define fully convolutional Generator network architecture
# This is a 6-layer CNN expecting a latent vector of size 128 as input
# This model emits a 3x128x128 tensor (an RGB color image)
#
generator = nn.Sequential(
	# input deconv layer:  tensor of LATENT_SIZEx1x1
	nn.ConvTranspose2d(LATENT_SIZE, 1024, kernel_size=4, stride=1, padding=0, bias=True),
	nn.BatchNorm2d(1024),
	nn.ReLU(True),

	# next deconv layer:  tensor of 1024x4x4
	nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=True),
	nn.BatchNorm2d(512),
	nn.ReLU(True),

	# next deconv layer:  tensor of 512x8x8
	nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
	nn.BatchNorm2d(256),
	nn.ReLU(True),

	# next deconv layer:  tensor of 256x16x16
	nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
	nn.BatchNorm2d(128),
	nn.ReLU(True),

	# next deconv layer:  tensor of 128x32x32
	nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
	nn.BatchNorm2d(64),
	nn.ReLU(True),

	# next deconv layer:  tensor of 64x64x64
	nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=True),
	nn.Tanh() )      # OUTPUT IS 3x128x128 Tensor




#
# Send our models to the GPU too!
#
if torch.cuda.is_available():
	print("Sending discriminator and generator to CUDA")
	discriminator.cuda()
	generator.cuda()

print("DISCRIMINATOR:")
summary(discriminator, input_size=(3,128,128))
print("\n\n")

print("GENERATOR:")
summary(generator, input_size=(LATENT_SIZE,1,1))
print("\n\n")


#
# Train the DISCRIMINATOR using the GENERATOR
#
# loss = real_loss + fake_loss
#
def train_discriminator(real_images, opt_d):

	# clear discriminator gradients
	opt_d.zero_grad()

	# Pass real images through discriminator
	real_preds   = discriminator(real_images)
	real_targets = torch.ones(real_images.size(0), 1, device=device)
	real_loss    = tfunc.binary_cross_entropy(real_preds, real_targets)
	real_score   = torch.mean(real_preds).item()

	# Generate fake images
	latent       = torch.randn(BATCH_SIZE, LATENT_SIZE, 1, 1, device=device)
	fake_images  = generator(latent)

	# Pass fake images through discriminator
	fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
	fake_preds   = discriminator(fake_images)
	fake_loss    = tfunc.binary_cross_entropy(fake_preds, fake_targets)
	fake_score   = torch.mean(fake_preds).item()

	# Update discriminator weights
	loss = real_loss + fake_loss
	loss.backward()
	opt_d.step()

	return loss.item(), real_score, fake_score



#
# Train the GENERATOR using the DISCRIMINATOR
#
def train_generator(opt_g):
	
	# clear generator gradients
	opt_g.zero_grad()

	# generate fake images
	latent = torch.randn(BATCH_SIZE, LATENT_SIZE, 1, 1, device=device)
	fake_images = generator(latent)

	# try to fool  the discriminator
	preds = discriminator(fake_images)
	targets = torch.ones(BATCH_SIZE, 1, device=device)
	loss = tfunc.binary_cross_entropy(preds, targets)

	# update generator weights
	loss.backward()
	opt_g.step()
	return loss.item()





##############################
#
#
# Main execution section
#
#
##############################

print("DISCRIMINATOR DEVICE: ", next(discriminator.parameters()).device)
print("GENERATOR DEVICE: ", next(generator.parameters()).device)


#
# Print something about our training data
#
for classes in os.listdir(INPUT_DIR):
	print(classes, ':', len(os.listdir(INPUT_DIR + '/' + classes)))


#
# Define our training dataset and perform transforms
#
train_dataset = ImageFolder(INPUT_DIR, 
			    transform=tt.Compose([tt.Resize(IMAGE_SIZE),      # resize image from 1024x1024 to 128x128
				        	  tt.CenterCrop(IMAGE_SIZE),  # crop center in case w != h
				        	  tt.ToTensor(),	      # convert image to Pytorch Tensor data structure
				        	  tt.Normalize(*STATS)]))     # image = (image - mean) / std


#
# Define a DataLoader based on our training set and BATCH_SIZE
#
train_dataloader = DataLoader(train_dataset,
			      BATCH_SIZE,
			      shuffle=True,
			      num_workers=NUM_WORKERS,
			      pin_memory=True)



#
# Make sure our dataloader send data to the GPU
#
device_train_dataloader = DeviceDataLoader(train_dataloader, device)


# Make a directory for image output
os.makedirs(OUTPUT_DIR, exist_ok=True)


#
# Generate a batch of faces and save a few to disk
#
def save_samples(index, latent_tensors, show=True):
	fake_images = generator(latent_tensors)
	fake_fname = 'generated-images-{0:0=6d}.png'.format(index)
	save_image(denormalize(fake_images[0:9]), os.path.join(OUTPUT_DIR, fake_fname), nrow=3)
	print('Saving', fake_fname)

#
# Generate a fixed latent vector of size LATENT_SIZE and use this to generate faces for saving to disk
#
fixed_latent = torch.randn(BATCH_SIZE, LATENT_SIZE, 1, 1, device=device)
save_samples(0, fixed_latent)    # save our first generated image to disk



#
# The main loop where we train (i.e. "fit" our models to our data)
#
def fit(epochs, lr, train_dl, start_idx=1):

	# clear out crap from our GPU
	torch.cuda.empty_cache()

	# Losses & scores
	losses_g = []
	losses_d = []
	real_scores = []
	fake_scores = []

	# Create Adam optimizers
	opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
	opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

	#
	# 
	#
	for epoch in range(epochs):
		for real_images, _ in tqdm(train_dl):

			# Train DISCRIMINATOR
			loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)

			# Train GENERATOR
			loss_g = train_generator(opt_g)

		# Record losses & scores
		losses_g.append(loss_g)
		losses_d.append(loss_d)
		real_scores.append(real_score)
		fake_scores.append(fake_score)

		# Log losses & scores (last batch)
		print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(epoch+1, epochs, loss_g, loss_d, real_score, fake_score))

		# Save generated images
		save_samples(epoch+start_idx, fixed_latent, show=False)

	return losses_g, losses_d, real_scores, fake_scores


print("BEGINNING TRAINING FOR %d EPOCHS..." %EPOCHS)

# TRAIN THE GAN
history = fit(EPOCHS, LEARNING_RATE, device_train_dataloader, start_idx=1)
losses_g, losses_d, real_scores, fake_scores = history

# Save the models when we are done training.  
# These trained models can be loaded again for re-training or generation.
torch.save(generator.state_dict(), 'G128.pth')
torch.save(discriminator.state_dict(), 'D128.pth')

print("DONE Training...")



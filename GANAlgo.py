import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import pandas as pd
import sklearn.utils

from DiscrimatorModel import Discriminator
from GeneratorModel import Generator

lr = 0.0002
b1 = 0.5
b2 = 0.999

n_epochs = 1 #200
batch_size = 32

latent_dim = 50

class DataBuffer:
    #HERE, SPLIT THE DATA UP INTO TRAIN AND TEST, SHUFFLE FOR EACH EPOCH, SPIT OUT BATCHES
    def __init__(self,batch_size):
        df = pd.read_csv('Data/processed_data.csv')
        
        #We are choosing doctor '4' to be the good one, who's diagnoses we are trying to model
        reliable_df = df.loc[df['doctors'] == 4]
        reliable_df = reliable_df.drop(['Unnamed: 0','doctors'], axis = 1)
        
        self.reliable_data = reliable_df.to_numpy()
        
        self.batch_size = batch_size
        self.num_batches = len(reliable_df) // batch_size
    
    def initialize(self):
        self.shuffled_reliable = sklearn.utils.shuffle(self.reliable_data)
        
    def get_batch(self, batch_num):
        return torch.tensor(self.shuffled_reliable[self.batch_size*batch_num:self.batch_size*(batch_num+1)], device = device, dtype = torch.float32)
        
        
        



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

adversarial_loss = nn.BCELoss().to(device)


optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr, betas = (b1,b2))

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = lr, betas = (b1,b2))

data = DataBuffer(batch_size)


for epoch in range(n_epochs):
    data.initialize()
    
    for i in range(data.num_batches):
        real_data = data.get_batch(i)
        label = torch.full((batch_size,),1.0,dtype = torch.float, device = device).view(-1)
        
        discriminator.zero_grad()
        output = discriminator(real_data).view(-1)
        errD_real = adversarial_loss(output,label)
        errD_real.backward()
        
        noise = torch.randn(batch_size, latent_dim, device = device)
        fake_data = generator(noise)
        label.fill_(0.0)
        errD_fake = criterion(output,label)
        errD_fake.backward()
        
        errD = (errD_real + errD_fake)/2
        
        optimizer_D.step()
        
        
        generator.zero_grad()
        label.fill_(1)
        output = discriminator(fake_data.detach()).view(-1)
        errG = criterion(output,label)
        
        errG.backward()
        optimizer_G.step()
        
        print("Epoch is " + str(epoch))
        print("Batch is " + str(i) + " out of " + str(data.num_batches))
        print("Discriminator loss is " + errD)
        print("Generator loss is " + errG)
              
              
        #ADD CODE TO SAVE MODELS AT THE END OF EACH EPOCH
        
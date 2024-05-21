import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from model import *
from NILM_dataset import *

class Trainer:
    def __init__(self, data, **kwargs):
        self.batch_size = kwargs["batch_size"]
        self.epochs = kwargs["epochs"]
        self.learning_rate = kwargs["learning_rate"]
        self.data = data
        self.input_shape = kwargs["input_shape"]

        self.train_loader = torch.utils.data.DataLoader(
            self.data, batch_size=self.batch_size, shuffle=kwargs["shuffle"]
)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoEncoder(input_shape=self.input_shape,
                            features=kwargs["num_features"]).to(self.device)
   
        self.features = kwargs["num_features"]

    def train(self):
        #  use gpu if available
        

        # create a model from `AutoEncoder` autoencoder class
        # load it to the specified device, either gpu or cpu
        

        # create an optimizer object
        # Adam optimizer with learning rate 1e-3
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # mean-squared error loss
        criterion = nn.MSELoss()
        seed = 42
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        
        for epoch in range(self.epochs):
            loss = 0
            for batch_features in self.train_loader:
                # reshape mini-batch data to [N, 784] matrix
                # load it to the active device
                batch_features = batch_features.view(-1, self.input_shape).to(self.device)
                
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()
                
                # compute reconstructions
                outputs = self.model(batch_features)
                
                # compute training reconstruction loss
                train_loss = criterion(outputs, batch_features)
                
                # compute accumulated gradients
                train_loss.backward()
                
                # perform parameter update based on current gradients
                optimizer.step()
                
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
            
            # compute the epoch training loss
            loss = loss / len(self.train_loader)
            
            # display the epoch training loss
            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, self.epochs, loss))
    
    def save(self, model_name):
        torch.save(self.model.state_dict(), model_name)

    def load(self, model_name):
        self.model = AutoEncoder(input_shape=self.input_shape,
                            features=self.features).to(self.device)

        self.model.load_state_dict(torch.load(model_name))
        self.model.eval()

    def visualize(self):

        with torch.no_grad():
            for batch_features in self.train_loader:
                test_examples = batch_features.view(-1, self.input_shape)
                reconstruction = self.model(test_examples)
                break

        with torch.no_grad():
            number = 10
            plt.figure(figsize=(20, 4))
            for index in range(number):
                # display original
                ax = plt.subplot(2, number, index + 1)
                plt.plot(test_examples[index].numpy())
                #plt.gray()
                #ax.get_xaxis().set_visible(False)
                #ax.get_yaxis().set_visible(False)

                # display reconstruction
                ax = plt.subplot(2, number, index + 1 + number)
                plt.plot(reconstruction[index].numpy())
                
                #plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()       

    def extract_features(self):
        features = []
        with torch.no_grad():
            for batch_features in self.train_loader:
                test_examples = batch_features.view(-1, self.input_shape)
                feature = self.model.feature_extraction(test_examples)

                features.append(feature)

            return features
    def extract_outputs(self):
        logits = []
        with torch.no_grad():
            for batch_features in self.train_loader:
                test_examples = batch_features.view(-1, self.input_shape)
                logit = self.model(test_examples)

                logits.append(logit)

            return logits


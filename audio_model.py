# ----------------------------
# Libaries
# ----------------------------
from utils import *
from audio_utility import *

import numpy as np
import pandas as pd
import torch, torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F




# ----------------------------------------------------------------------------------------------------------------
# Sound Dataset
# ----------------------------------------------------------------------------------------------------------------
class  Sound_DataLoader(Dataset):
    """Generates data for Torch
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, df, target = "emotion", audio_path = "full_path", feature = None):
        self.df         = df
        self.duration   = 3500
        self.sr         = 44100
        self.channel    = 2
        self.shift_pct  = 0.4
        
        self.audio_path = audio_path
        self.target     = target
        self._audio     = AudioUtility()
        
        self.features   = feature

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)    

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, index):
        if self.features is not None: 
            feature    = self.df.loc[index, self.features].tolist()
            feature = torch.FloatTensor(feature)

        audio_file = self.df.loc[index, self.audio_path] # Absolute file path of the audio file
        class_id   = self.df.loc[index, self.target]     # Get the Class ID

        audio = self._audio.open(audio_file)
        audio = self._preprocess(audio)

        if self.features is not None: return audio, feature, class_id
        else:                         return audio, class_id

    def _preprocess(self, audio):
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same 
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        audio  = self._audio.resample(audio, self.sr)
        audio  = self._audio.rechannel(audio, self.channel)
        audio  = self._audio.pad_trunc(audio, self.duration)
        audio  = self._audio.time_shift(audio, self.shift_pct)
        audio  = self._audio.spectro_gram(audio, n_mels=64, n_fft=1024, hop_len=None)
        audio  = self._audio.spectro_augment(audio, max_mask_pct=0.04, n_freq_masks=4, n_time_masks=4)

        return audio
    
    

    
    
    

    
    
    



# ----------------------------------------------------------------------------------------------------------------
# Audio Classification Model
# ----------------------------------------------------------------------------------------------------------------
class AudioClassifier (torch.nn.Module):
    def conv_block(self, input_size, output_size, kernel_size, padding):
        conv = torch.nn.Conv2d(input_size, output_size, (kernel_size, kernel_size), stride=(2, 2), padding=(padding, padding))
        relu = torch.nn.ReLU()
        bnor = torch.nn.BatchNorm2d(output_size)
        
        torch.nn.init.kaiming_normal_(conv.weight, a=0.1)
        conv.bias.data.zero_()
        
        block = torch.nn.Sequential(conv, relu, bnor)

        return block
    

    def __init__(self):
        """
        In the constructor we instantiate parameters and assign them as
        member parameters.
        """
        super().__init__()
        self._init_cnn_()
        self._init_fea_()
        self._init_cat_()
        

    def forward(self, img, features = None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # Run the convolutional blocks
        #self._cnn_forward_()
                
        conv_layers = []
        conv_layers += [self.conv1]
        conv_layers += [self.conv2]
        conv_layers += [self.conv3]
        conv_layers += [self.conv4]
        conv_layers += [self.conv5]
        conv_layers += [self.conv6]
        self.conv = torch.nn.Sequential(*conv_layers)
        
        
        img = self.conv(img)

        # Adaptive pool and flatten for input to linear layer
        img = self.ap(img)
        img = img.view(img.shape[0], -1)
        
        
        ######
        if features is not None:
            features = self.lin1(features)
            features = self.relu(features)
            features = self.lin2(features)
            features = self.relu(features)
            features = self.lin3(features)
            features = self.relu(features)
        
            #####
            x = torch.cat((img, features), dim=1)
            x = self.relu(x)
        
            x = self.lin4(x)
            x = self.lin5(x)
            x = self.lin6(x)
        else:
            x = self.lin(img)
        
        
        # Final output
        return x    
    
    def _init_cnn_(self):
        self.conv1 = self.conv_block(input_size = 2,  output_size = 8,  kernel_size = 5, padding = 2)
        self.conv2 = self.conv_block(input_size = 8,  output_size = 16, kernel_size = 3, padding = 1)
        self.conv3 = self.conv_block(input_size = 16, output_size = 32, kernel_size = 3, padding = 1)
        self.conv4 = self.conv_block(input_size = 32, output_size = 64, kernel_size = 3, padding = 1)
        self.conv5 = self.conv_block(input_size = 64, output_size = 128, kernel_size = 3, padding = 1)
        self.conv6 = self.conv_block(input_size = 128, output_size = 64, kernel_size = 3, padding = 1)
        conv_layers = []
        conv_layers += [self.conv1]
        conv_layers += [self.conv2]
        conv_layers += [self.conv3]
        conv_layers += [self.conv4]
        conv_layers += [self.conv5]
        conv_layers += [self.conv6]
        self.conv = torch.nn.Sequential(*conv_layers)
        
        # Linear Classifier
        self.ap = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = torch.nn.Linear(in_features=64, out_features=8)
        
    def _init_fea_(self):
        self.lin1 = torch.nn.Linear(in_features=4,  out_features=10)
        self.lin2 = torch.nn.Linear(in_features=10, out_features=10)
        self.lin3 = torch.nn.Linear(in_features=10, out_features=10)
        
        self.relu = torch.nn.ReLU()
    
    def _init_cat_(self):
        self.lin4 = torch.nn.Linear(in_features=64+10, out_features=32)
        self.lin5 = torch.nn.Linear(in_features=32,    out_features=32)
        self.lin6 = torch.nn.Linear(in_features=32,    out_features=8)



    
    
    
    


    
    
    
    
# ----------------------------------------------------------------------------------------------------------------
# Speech Model
# ----------------------------------------------------------------------------------------------------------------

class speech_model:
    def __init__(self, model, epochs = 10, verbose = 0, device = None):
        self.model            = model
        self.epochs           = epochs
        self.verbose          = verbose
        self._epoch_frequency = 10
        self.device = device
        
    def fit(self, training_dataloader, validation_dataloader = None):
        
        if self.verbose > 0: print2.highlight("Fitting model")
        
        self.training_dataloader = training_dataloader
        self.train_length        = len(self.training_dataloader)
        

        
            # Loss Function, Optimizer and Scheduler
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.001,
                                                steps_per_epoch=int(self.train_length),
                                                epochs=self.epochs,
                                                anneal_strategy='linear')
        
        min_valid_acc = 0
        # Repeat for each epoch
        for epoch in range(self.epochs):
            self.correct_prediction = 0
            self.total_prediction   = 0
            train_loss              = 0.0
            
            self.model.train()
            # Repeat for each batch in the training set
            for i, data in enumerate(self.training_dataloader):
                # Get the input features and target labels, and put them on the GPU, if available
                if len(data) == 2:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                elif len(data) == 3:
                    inputs, inputs2, labels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                else:
                    break

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                self.optimizer.zero_grad()
                
                if len(data) == 3:
                    outputs    = self.model(inputs, inputs2)
                else:
                    outputs    = self.model(inputs)
                
                loss       = self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None: self.scheduler.step()
                train_loss += loss.item()
                
                accuracy   = self._accuracy(outputs, labels)
                
                
                self._print_batch(current_epoch = epoch, current_batch = i, batch_length = self.train_length, 
                                  accuracy = accuracy,
                                  verbose = self.verbose)
            if validation_dataloader is not None: valid_acc  = self._evaluate(validation_dataloader)
            else: valid_acc = None
                
            if min_valid_acc < valid_acc:
                torch.save(self.model.state_dict(), 'saved_model.pth')
            
            self._print_epoch(current_epoch = epoch, loss = train_loss / len(self.training_dataloader), 
                              accuracy = accuracy, validation_accuracy = valid_acc,
                              verbose = self.verbose)
            
            #if self.scheduler is not None: self.scheduler.step()
        print("", end = "\n")

                
    def _evaluate(self, validation_dataloader, verbose = 0):
        self.correct_prediction = 0
        self.total_prediction   = 0
        valid_loss = 0.0
        self.model.eval()
        
        run_acc = 0
        for i, data in enumerate(validation_dataloader):
            # Get the input features and target labels, and put them on the GPU, if available
            if len(data) == 2:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
            elif len(data) == 3:
                inputs, inputs2, labels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
            else:
                break
                    
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            if len(data) == 3:
                outputs    = self.model(inputs, inputs2)
            else:
                outputs    = self.model(inputs)
                
            loss       = self.criterion(outputs,labels)
            valid_loss = loss.item() * inputs.size(0)

            accuracy   = self._accuracy(outputs, labels)
            run_acc    += accuracy 
            
            if verbose > 0: 
                self._print_validation(current_batch = i, batch_length = len(validation_dataloader),
                                       validation_accuracy = run_acc/(i+1), verbose = self.verbose)

            
        return (run_acc/len(validation_dataloader))

    
    def evaluate(self, validation_dataloader):
        if self.verbose > 0: print2.highlight("Evaluating")
        self.validation_dataloader = validation_dataloader
        self.valid_length          = len(self.validation_dataloader)
        
        correct_prediction = 0
        total_prediction   = 0

        valid_loss = 0.0
        #self.model.eval()

        self.correct_prediction = 0
        self.total_prediction   = 0
            
        accuracy = self._evaluate(validation_dataloader, verbose = self.verbose)

        
    def predict(self, test_dataloader, return_all = False):
        outputs = []
        self.model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                self._print_prediction(current_batch = i, batch_length = len(test_dataloader), verbose = self.verbose)
                
                    # Get the input features and target labels, and put them on the GPU, if available
                if len(data) == 2:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                elif len(data) == 3:
                    inputs, inputs2, labels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                else:
                    break

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                if len(data) == 3:
                    output    = self.model(inputs, inputs2)
                else:
                    output    = self.model(inputs)
                
                
                if not return_all: 
                    _, output = torch.max(output,1)
                    output = output.numpy()
                elif return_all:
                    output = torch.round(torch.softmax(output,1),decimals = 3)
                    output = output.numpy()
                
                outputs.append(output)

        if not return_all: outputs = np.array(outputs)  
        return outputs



    def _accuracy(self, outputs, labels):
        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
            
        # Count of predictions that matched the target label
        self.correct_prediction += (prediction == labels).sum().item()
        self.total_prediction += prediction.shape[0]

        accuracy = self.correct_prediction/self.total_prediction
        return(accuracy)
    
    def _print_epoch(self, current_epoch, loss = None, validation_loss = None, accuracy = None, validation_accuracy = None,
                     verbose = 0, frequency = 10, width = 125):
        
        # If verbose == 0, then print nothing. If verbose == 1, then print every now and then (based on frequency)
        #  however, if verbose == 2 (or not 0, 1), then print every epoch.
        if verbose   == 0: return
        elif verbose == 1: 
            if  (current_epoch+1) % frequency  == 0:           end = "\n"
            elif current_epoch+1               == 1:           end = "\n"
            elif current_epoch+1               == self.epochs: end = "\n"
            else:                                              end = "\r"
        else:                                                  end = "\n"
        
        _word   = print2.bold("Epoch:", _return = True, _print = False)
        _string = _word + f" {current_epoch+1} of {self.epochs}\t"
        
        if loss is not None:
            _word    = print2.bold("Loss:", _return = True, _print = False)
            _string += "\t" + _word +  f" {loss:.3f}"
        if validation_loss is not None:
            _word    = print2.bold("Validation Loss:", _return = True, _print = False)
            _string += "\t" + _word +  f" {validation_loss:.3f}"
        if accuracy is not None:
            _word    = print2.bold("Accuracy:", _return = True, _print = False)
            _string += "\t" + _word +  f" {(accuracy*100):.2f}%"
        if validation_accuracy is not None:
            _word    = print2.bold("Validation Accuracy:", _return = True, _print = False)
            _string += "\t" + _word +  f" {(validation_accuracy*100):.2f}%"
        
        _padding = " " * max(1, (width - len(_string)))
        print(_string + _padding, end = end)
        
        
    def _print_batch(self, current_epoch, current_batch = None, batch_length = None, accuracy = None, 
                     validation_accuracy = None, verbose = 0, width = 125):        
        if verbose   == 0: return
        elif verbose == 1: end = "\r"
        elif verbose == 2: end = "\r"
        else:            end = "\n"
         
        
        _string = ""
        if current_epoch is not None:
            _word   = print2.bold("Epoch:", _return = True, _print = False)
            _string = _word + f" {current_epoch+1} of {self.epochs}\t"
        
        if (current_batch is not None) or (batch_length is not None): 
            _word    = print2.bold("Batch:", _return = True, _print = False)
            _string += "\t" + _word +  f" {current_batch+1} of {batch_length}"
        
        if accuracy is not None:
            _word    = print2.bold("Accuracy:", _return = True, _print = False)
            _string += "\t" + _word +  f" {accuracy*100:.2f}%"
        if validation_accuracy is not None:
            _word    = print2.bold("Validation Accuracy:", _return = True, _print = False)
            _string += "\t" + _word +  f" {validation_accuracy*100:.2f}%"  
        
        _padding = " " * (width - len(_string))
        print(_string + _padding, end = end)

    def _print_validation(self, current_batch = None, batch_length = None, 
                          accuracy = None, validation_accuracy = None, 
                          verbose = 0, width = 125):  
        
        if verbose   == 0: return
        elif verbose == 1: end = "\r"
        elif verbose == 2: end = "\r"
        else:            end = "\n"
         
        _string = ""
        
        if (current_batch is not None) or (batch_length is not None): 
            _word    = print2.bold("Item:", _return = True, _print = False)
            _string += _word +  f" {current_batch+1} of {batch_length}"
        
        if accuracy is not None:
            _word    = print2.bold("Mean Accuracy:", _return = True, _print = False)
            _string += "\t" + _word +  f" {accuracy*100:.2f}%"
        if validation_accuracy is not None:
            _word    = print2.bold("Mean Validation Accuracy:", _return = True, _print = False)
            _string += "\t" + _word +  f" {validation_accuracy*100:.2f}%"  
        
        _padding = " " * (width - len(_string))
        print(_string + _padding, end = end)
    
    def _print_prediction(self, current_batch = None, batch_length = None, verbose = 0, width = 125):
        if verbose   == 0: return
        elif verbose == 1: end = "\r"
        elif verbose == 2: end = "\r"
        else:            end = "\n"
         
        _string = ""
        
        if (current_batch is not None) or (batch_length is not None): 
            _word    = print2.bold("Predicting Item:", _return = True, _print = False)
            _string += _word +  f" {current_batch+1} of {batch_length}"
            
        _padding = " " * (width - len(_string))
        print(_string + _padding, end = end)

        
        

        
 
    
    
    


    
    
    
    
# ----------------------------------------------------------------------------------------------------------------
# One Off Prediction via file path
# ----------------------------------------------------------------------------------------------------------------
class speech_prediction():
    def __init__(self, model = "saved_model.pth"):
        self.model  = AudioClassifier()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model  = self.model.to(self.device)
        self.model.load_state_dict(torch.load("saved_model.pth"))
        
        self.sm     = speech_model(model = self.model, device = self.device)
        
        self.audio_util = AudioUtility()

    def predict(self, audio_file, return_all = False):
        with torch.no_grad():
            self.model.eval()
            audio = self.audio_util.open(audio_file)
            audio = self.audio_util.resample(audio, 44100)
            audio = self.audio_util.rechannel(audio, 2)
            audio = self.audio_util.pad_trunc(audio, 3500)
            audio = self.audio_util.spectro_gram(audio, n_mels=64, n_fft=1024, hop_len=None)

            # Normalize the inputs
            inputs_m, inputs_s = audio.mean(), audio.std()
            inputs = (audio - inputs_m) / inputs_s

            if len(inputs.size()) == 3: inputs = inputs[None, :,:,:]
            prediction = self.model(inputs)
        
            if not return_all: 
                _, output = torch.max(prediction,1)
                output = output.numpy()
            elif return_all:
                output = torch.round(torch.softmax(prediction,1),decimals = 3)
                output = output.numpy()[0]

            if not return_all: 
                emotion_lookup = {0: "neutral", 1: "calm", 2: "happy", 3: "sad", 4: "angry",
                      5: "fearful", 6: "disgust", 7: "surprised"}

                output = emotion_lookup.get(int(output))

        return output
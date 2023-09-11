from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import PostPre
from transformers import DistilBertModel, DistilBertTokenizer
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from autoencoder import Autoencoder
from rum import RUMNodes
# Connection from hidden to output layer with MSTDP learning rule and rates
from bindsnet.learning import MSTDP


class SpikingTransformerNetwork(nn.Module):
    def __init__(self, input_dim=544, hidden_dim=1024, output_dim=2048, dt=1.0):  
        super(SpikingTransformerNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context = torch.zeros(1,output_dim)
        self.squeezing_layer = nn.Linear(input_dim, output_dim)  # Add this line to initialize the new layer

        self.dt = dt
        # Initialize DistilBERT
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # Initialize BindsNET network components
        self.network = Network()
        #self.inp = Input(n=input_dim)
        #self.hidden = LIFNodes(n=hidden_dim, rest=-65.0, reset=-65.0, thresh=-52.0)
        #self.out = LIFNodes(n=output_dim, rest=-65.0, reset=-65.0, thresh=-52.0)
        # More human-like parameters
        self.inp = LIFNodes(n=input_dim, rest=-70, reset=-65, thresh=-55, tau=10.0, traces=True)
        self.hidden = RUMNodes(n=hidden_dim, dt=dt, alpha=0.95, beta=0.8, traces=True)
        self.out = LIFNodes(n=output_dim, rest=-70, reset=-65, thresh=-55, tau=10.0, traces=True)
        # Initialize Autoencoder
        self.autoencoder = Autoencoder(input_dim=768, encoding_dim=input_dim)
        
        self.network.add_layer(self.inp, name='inp')
        self.network.add_layer(self.hidden, name='hidden')
        self.network.add_layer(self.out, name='out')

        # Connection from input to hidden layer with learning rule and rates
        self.input_to_hidden = Connection(
            source=self.inp,
            target=self.hidden,
            update_rule=PostPre,
            nu=(1e-3, 1e-2),
            wmin=0.0,  # minimum weight value
            wmax=1.0  # maximum weight value
        )
        self.network.add_connection(self.input_to_hidden, source='inp', target='hidden')

        # Connection from hidden to output layer with learning rule and rates
        self.hidden_to_output = Connection(
            source=self.hidden,
            target=self.out,
            update_rule=MSTDP,
            nu=(1e-4, 1e-2)  # You can tune these rates
        )

        self.network.add_connection(self.hidden_to_output, source='hidden', target='out')

        # Create and add monitors
        input_monitor = Monitor(self.inp, ['s'], time=500)
        hidden_monitor = Monitor(self.hidden, ['s'], time=500)
        output_monitor = Monitor(self.out, ['s'], time=500)

        self.network.add_monitor(input_monitor, name='input_monitor')
        self.network.add_monitor(hidden_monitor, name='hidden_monitor')
        self.network.add_monitor(output_monitor, name='output_monitor')

    def forward(self, x, num_timesteps=500, reward=1.0):
        # Debugging: Print layer sizes
        for name, param in self.network.named_parameters():
            print(f'Layer: {name}, Size: {param.size()}')


        print("Shape of self.input_to_hidden.w:", self.input_to_hidden.w.shape)
        print("Shape of self.context:", self.context.shape)

        # Modulating Connection Weights
        # Say the first 256 weights in self.input_to_hidden.w are relevant for the current context
        self.input_to_hidden.w[:, :256] *= (1 + self.context)

        
        print("Shape of self.inp.rest:", self.inp.rest.shape)
        print("Shape of self.hidden.rest:", self.hidden.rest.shape)

        # Influencing Spiking Behavior
        self.inp.rest = self.inp.rest + self.context
        self.hidden.rest = self.hidden.rest + self.context

        # Step 2: Apply rate coding on the encoded data
        encoded_x = self.rate_coding(x, num_timesteps, self.dt)

        # Debugging: Print the encoded tensor
        print("Encoded X:", encoded_x)
        
        print("input rest: ", self.inp.rest, "input reset: ", self.inp.reset, "input threshold: ", self.inp.thresh)
        print("hidden alpha: ",self.hidden.alpha, "hidden beta:",self.hidden.beta)
        print("out rest: ", self.out.rest, "out reset:",self.out.reset, "out threshold",self.out.thresh)

        print("Connection weights [input to hidden]: ", self.input_to_hidden.w)
        print("Connection weights [hidden to output]: ", self.hidden_to_output.w)

        # Initialize a list to hold spikes at each timestep
        all_spikes = []

        for t in range(num_timesteps):
            print("Shape of encoded_x before self.network.run:", encoded_x.shape)
            # Run the spiking neural network for one timestep using the encoded data
            # dummy_input = torch.randn(500, 1, 800)
            self.network.run(inputs={'inp': encoded_x}, reward=torch.tensor(reward, dtype=torch.float), time=1)

            # Get the monitors
            input_monitor = self.network.monitors.get('input_monitor')
            hidden_monitor = self.network.monitors.get('hidden_monitor')
            output_monitor = self.network.monitors.get('output_monitor')

            # Debugging: Print spikes
            print(f"Time {t}:")
            print("Input spikes:", input_monitor.recording['s'][t])
            print("Hidden spikes:", hidden_monitor.recording['s'][t])
            print("Output spikes:", output_monitor.recording['s'][t])

            # Get the spikes at this timestep for the output layer
            spikes = output_monitor.recording['s'][t]

            # Append the spikes to the list
            all_spikes.append(spikes)

        # When you stack the spikes
        print("Individual spike tensor shapes:", [s.shape for s in all_spikes])
        all_spikes_tensor = torch.stack(all_spikes)
        print("Shape of all_spikes_tensor:", all_spikes_tensor.shape)

        
        self.update_context(all_spikes_tensor.float().mean(dim=0))
        
        return all_spikes_tensor

    def encode_features(self, data_vector):
        # Call encode once
        print("Shape of data_vector:", data_vector.shape)
        encoded_data = self.autoencoder.encode(data_vector)
        print("Shape of data_vector_reduced:", encoded_data.shape)
        return encoded_data


    def update_context(self, data_vector):
        decay_factor = 0.9
        print("Shape of self.context:", self.context.shape)
        # Use the new layer to adjust the dimensions
        squeezed_data_vector = self.squeezing_layer(data_vector)

        # Make sure it's of the same shape as self.context, which is [32]
        squeezed_data_vector = torch.squeeze(squeezed_data_vector)
        
       
        self.context = decay_factor * self.context + (1 - decay_factor) * squeezed_data_vector

    def extract_features(self, text):
        print("Type of text:", type(text))
        print("Value of text:", text)
        actual_text = text.get('tokens', {}).get('new_words', [])
        if actual_text:
            actual_text = ' '.join(actual_text)  # Convert list of words to a single string
        else:
            print("No text found in the dictionary.")
            return
        # Now use the tokenizer
        try:
            inputs = self.tokenizer(actual_text, return_tensors="pt", padding=True, truncation=True)
        except Exception as e:
            print(f"An error occurred: {e}")
        outputs = self.distilbert(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Taking the mean of the sequence dimension as a simple pooling strategy
    
    def rate_coding(self, input_vector, time, dt):
        # Ensure input_vector is a 2D tensor of shape [1, n]
        rates = torch.Tensor(input_vector).view(1, -1)
        print(f"Shape of rates: {rates.shape}")  # Debug

        # Reshape self.context to [1, 32]
        context_reshaped = self.context.view(1, -1)
        print(f"Shape of context_reshaped: {context_reshaped.shape}")  # Debug

        # Concatenate rates and self.context along dim=1
        concatenated_rates = torch.cat((rates, context_reshaped), dim=1)
        print(f"Shape of concatenated_rates: {concatenated_rates.shape}")  # Debug

        # Create the time steps
        time_steps = torch.arange(0, time, dt)
        print(f"Number of time_steps: {len(time_steps)}")  # Debug

        # Expand the concatenated tensor along the time dimension
        encoded = concatenated_rates.unsqueeze(0).expand(len(time_steps), -1, -1)
        print(f"Shape of encoded: {encoded.shape}")  # Debug

        return encoded

    
        '''def rate_coding(self, input_vector, time, dt):
        rates = torch.Tensor(input_vector)
        
        # As an Additional Input
        encoded = torch.cat((rates, self.context), dim=1).unsqueeze(0).expand(len(time_steps), -1, -1)

        return encoded'''
    
# Example usage
if __name__ == "__main__":
    stn = SpikingTransformerNetwork(input_dim=768, hidden_dim=64, output_dim=32)
    text = "This is a sample sentence."
    features = stn.extract_features(text)
    output_data = stn(features)
    print(f"Output data: {output_data}")
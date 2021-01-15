"""

    Created on 13/01/21 5:52 PM 
    @author: Kartik Prabhu

"""
from utils import init_weights, embedding
import numpy as np
import Layers

class Network:

    # def update_weights(self, index, gradient):
    #       self.w[index].assign_sub(self.learning_rate * gradient)

    # def update_bias(self, index, gradient):
    #       self.b[index].assign_sub(self.learning_rate * gradient)

    def __init__(self, input_dim,encoder_hidden_dim,decoder_hidden_dim,output_dim, weight_init_type = 'uniform'):
        self.input_dim = input_dim
        self.hidden_dim = encoder_hidden_dim
        self.output_dim = output_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        self.loss = None

        # Weights and biases for encoder
        self.input_weights = init_weights(weight_init_type,0, 1,self.input_dim,self.hidden_dim)
        self.hidden_state_weights = init_weights(weight_init_type, 0, 1, self.hidden_dim, self.hidden_dim)
        self.output_weights = init_weights(weight_init_type,0,1,self.hidden_dim,self.output_dim)

        self.b = np.zeros(self.hidden_dim, dtype=np.float32)# bias for hidden
        self.c = np.zeros(self.output_dim, dtype=np.float32) # bias for output

        # delta values
        self.dU = np.zeros_like(self.input_weights)
        self.dV = np.zeros_like(self.output_weights)
        self.dW = np.zeros_like(self.hidden_state_weights)

        # self.sigmoid = Layers.Sigmoid()
        self.tanh = Layers.Tanh()

        # model(decoder)
        self.dec_input_weights =  init_weights(weight_init_type,0, 1,self.input_dim,self.decoder_hidden_dim)
        self.dec_hidden_state_weights =  init_weights(weight_init_type,0, 1,self.decoder_hidden_dim,self.decoder_hidden_dim)
        self.dec_output_weights = init_weights(weight_init_type,0,1,self.decoder_hidden_dim,self.output_dim)
        self.dec_b = np.zeros(self.decoder_hidden_dim, dtype=np.float32)# bias for hidden
        self.dec_c = np.zeros(self.output_dim, dtype=np.float32) # bias for output

        self.parameters = [self.b, self.c, self.input_weights, self.hidden_state_weights, self.output_weights,
                           self.dec_b, self.dec_c, self.dec_input_weights, self.dec_hidden_state_weights, self.dec_output_weights]


    def getParameters(self):
            return self.parameters

    '''
    x: batch of input
    '''
    def forward(self, input_batch, target_batch):
        # print(input_batch.shape)
        # print(self.hidden_state_weights.shape)

        hidden_state = np.zeros((input_batch.shape[0], self.hidden_dim))
        decoder_state = np.zeros((target_batch.shape[0], self.decoder_hidden_dim))
        #encode
        for i in range(0, input_batch.shape[1]):
            a_t = self.b + np.matmul(input_batch[:, i], self.input_weights) + np.matmul(hidden_state, self.hidden_state_weights)
            #(1,512) + (1,68)(68,512) + (1,512)(512,512)
            hidden_state = self.tanh.forward(a_t)
            # logits = self.c + np.matmul(hidden_state, self.hiddent_state_weights) #(1,68)+(1,512)(512,68)
            decoder_state = hidden_state

        dec_input = target_batch[:, 0]
        #decode
        for i in range(0, target_batch.shape[1]):
            # print("===========")
            # print(self.dec_b.shape)
            # print(dec_input[:, i].shape)
            # print(hidden_state.shape)
            # print(self.input_weights.shape)
            # print(self.output_weights.shape)

            a_t_dec = self.dec_b + np.matmul(dec_input, self.dec_input_weights) + np.matmul(decoder_state, self.dec_hidden_state_weights) #(1,512) + (1,68)(68,512) + (1,512)(512,512)
            decoder_state = self.tanh.forward(a_t_dec)
            logits = self.dec_c + np.matmul(decoder_state, self.dec_output_weights) #(1,68)+(1,512)(512,68)

            dec_input = target_batch[:, i]

        return logits
        # xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=out_char))

    def backward(self, x, outputGrad):
            self.gradWeight = np.dot(x.T, outputGrad)
            self.gradBias = np.copy(outputGrad)
            return np.dot(outputGrad, self.weight.T)


    # def calculateLoss(self):

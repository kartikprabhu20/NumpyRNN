import numpy as np
from utils import init_weights, embedding
from Network import Network
from Dataset import Dataset

if __name__ == '__main__':
    hidden_dim = 100
    learning_rate = 0.0001
    epochs = 25
    length_of_sequence = 10
    max_no = 20
    input_dim = max_no # since we do one_hot encoding
    output_dim = input_dim
    batch_size = 4
    total_dataset = 20

    dataset = Dataset(length_of_sequence,batch_size,max_no,total_dataset)
    net = Network(input_dim,hidden_dim, hidden_dim,output_dim)

    for i in range(0,epochs):
        print("Epoch:"+ str(i))
        for i in range (int(total_dataset/batch_size)):
            inputX,outputY = dataset.getBatch(i)
            # print(inputX)
            # print(outputY)

            #Convert to one_hot encoding
            inputX_oh = embedding(inputX, length_of_sequence, max_no)
            outputY_oh = embedding(outputY, length_of_sequence, max_no)
            # print(inputX_oh)
            # print("============")
            # print(inputX[:,1])
            # print(inputX_oh[:,0])
            net.forward(inputX_oh,outputY_oh)








        # h_t = tf.zeros([1,net.hidden_units])
        # # print(h_t)  #(1,512)
        #
        # for char_index in range(0, len(current_batch)-1):
        #     # print("char_index = "+ str(char_index))
        #     input_char = oh_batch[vector_index][char_index]
        #     input_char = tf.reshape(input_char, [1,len(input_char)])
        #     # print(input_char) #(1,68)
        #
        #     out_char = oh_batch[vector_index][char_index + 1]
        #     out_char = tf.reshape(out_char, [1,len(out_char)])
        #     # print(out_char) #(1,68)
        #
        #     xent, logits = train_step(h_t,input_char, out_char)
        #     # print(logits)
        #
        #     if not batch_number % 5:
        #         preds = tf.argmax(logits, axis=-1)
        #         # print(preds)
        #         # print(xent)
        #         out_char = tf.argmax(out_char,axis=-1)
        #         # print(out_char)
        #         acc = tf.reduce_mean(tf.cast(tf.equal(preds, out_char),tf.int32))
        #         print("Batch:{}".format(batch_number)+" Loss: {} Accuracy: {}".format(xent, acc))




    test_sequence = np.array([[2,5,6,7,1,9,8]])
    assert np.max(test_sequence) <= max_no
    length_of_sequence = len(test_sequence[0])
    test = embedding(test_sequence, length_of_sequence, max_no) #max_no should be same as training irrespective of the seq_length
    # print(test_sequence)
    # print(test)


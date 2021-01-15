"""

    Created on 15/01/21 11:19 AM 
    @author: Kartik Prabhu

"""
import numpy as np

class Dataset():

    def __init__(self, length_of_sequence,batch_size,max_no,total_dataset):
        self.length_of_sequence =length_of_sequence
        self.batch_size = batch_size
        self.max_no =max_no
        self.total_dataset = total_dataset

        self.data= np.random.randint(max_no, size=(total_dataset,length_of_sequence))

        # print(self.data)

    def getBatch(self,index):
        batch = self.data[index * self.batch_size : index * self.batch_size + self.batch_size ]
        batch_output = np.sort(batch)
        return batch, batch_output



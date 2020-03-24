import time
import math
import os
import json
from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import argparse
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib import pyplot

class temp_device(object):
    """Represents the state of a single device."""
    def __init__(self):
        self.temperature = 0
        self.lamp_on = False
        self.curve_flat = True
        self.connected = False
        self.temps = np.zeros((1,360,2)) #two features air and out temp
    def log_air_temp(self,index,air,time):
        self.temps[0,index,0] = air
        self.temps[0,index,1] = time
    def start_temp(self):
        self.lamp_on = True
        print("Lamp On")

def upload_blob(bucket_name, blob_text, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(blob_text)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

def log_data(request):
    request_json = request.get_json()
    BUCKET_NAME = 'iot_bucket_453'
    BLOB_NAME = 'test-blob'
    BLOB_STR = '{"blob": "some json"}'

    upload_blob(BUCKET_NAME, BLOB_STR, BLOB_NAME)
    return f'Success!'

###
#Takes an input directory and tries to append current dir to it
#Checks if the input directory is a directory
###
def dir_path(string):
    string2 = os.getcwd() + string
    if os.path.isdir(string2):
        return string2
    elif os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
###
#Implements command line arguemnts
#-e is number of epochs
#-tp is file path point to test data
#-vp is file path point to val data
#-o is a output file which model will be saved too
#-i is a input file which model is loaded from 
#-vcol is the col with truth/val data
#-stcol is for temp and time data
#-srcol is for data which changes on a per run basis 
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_file', help='Input file for Network weights',required=False)
    return parser.parse_args()


def main():
    scalers = {}
    local_batch_size=18
    with open('turnkey-banner-265721-da1327341af6.json', 'r') as json_file:
        data = json.load(json_file)
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(
        data
    )
    data_log = 'run0.txt'
    read_log = 'data_2025100.csv' 
    device = temp_device()
    input_string = ""
    with open(read_log,"r") as read_file:
        lines = [line.rstrip() for line in read_file]
    while True: 
        val_data = np.zeros((360,1,2))

        time.sleep(1)
        print("Start")
        usrInputMsg=input()
        print(usrInputMsg)
        if(usrInputMsg == "Y"):
            print("Y")
            device.start_temp()
            with open(data_log, "w") as write_file:
                start_time = time.time()
                index = 0
                for i in range(0,len(lines)-10,10):
                    index += 1
                    line = lines[i] 
                    #time.sleep(.01)
                    print("Time: "+str(time.time() - start_time))
                    #input_string = '{"Type": 0, "Data" :'+str(line)+', "Time" : '+str(time.time() - start_time)+'}'
                    #print(input_string)
                    input_string = json.dumps({"Type": 0, "Data" :str(line), "Time" : str(time.time() - start_time)})
                    
                    print(input_string)
                    output_string = json.loads(input_string)
                    #print(json.load(paylod))
                    if(output_string["Type"] == 0):
                        wr_data = output_string["Data"]
                        split_write_data = wr_data.split(",")
                        t_data = output_string["Time"] 
                        write_file.write(wr_data+","+t_data+"\n")
                        print("Time: "+split_write_data[0])
                        print("Temp: "+split_write_data[1])
                        device.log_air_temp(index,split_write_data[0],split_write_data[1])
                        val_data[index,0,0] = split_write_data[0]
                        val_data[index,0,1] = split_write_data[1]
            write_file.close()     
            parsed_args = parse_arguments()
            in_file_name = parsed_args.in_file

            model = load_model(in_file_name)    
       

            in_file_name = parsed_args.in_file
            model = load_model(in_file_name)    

            val_scaled = val_data
            val_data_bak = val_data
            #Scale data that change during run this way
            for j in range(val_scaled.shape[0]):    
                scalers[0] = MinMaxScaler(feature_range=(0, 1))
                scalers[1] = MinMaxScaler(feature_range=(0, 1))
                val_scaled[:,0, 0:1] = scalers[0].fit_transform(val_data[:,0,0:1])  
                val_scaled[:,0, 1:2] = scalers[1].fit_transform(val_data[:,0,1:2])
            print(val_scaled[0:50,:,:])
            yhat = model.predict(val_scaled[:,0:1,:] , batch_size = local_batch_size)
            val_scaler = MinMaxScaler(feature_range=(0,1)).fit(val_data_bak[:,0,1:2])
            inv_yhat_out = val_scaler.inverse_transform(yhat)
            print(yhat)
            with open("yhat.txt", "w") as yhat_file:
                for line in yhat:
                    yhat_file.write(str(line) + "\n")



            # client = storage.Client(credentials=credentials, project='turnkey-banner-265721')
            # bucket = client.get_bucket('iot_bucket_453')
            # blob = bucket.blob('test-file2')
            # blob.upload_from_filename(data_log)
if __name__ == '__main__':
    main()
